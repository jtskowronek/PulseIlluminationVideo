import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Tim
import numpy as np
from timm.models.layers import trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from config.utils.utils import A, At
import bisect


def inference(Demodel, img_batch_1, img_batch_2, inter_frames):
        results = [
            img_batch_1,
            img_batch_2
        ]

        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)

        for _ in range(len(remains)):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            x0 = x0.cuda()
            x1 = x1.cuda()

            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = Demodel(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]
        return results





class cnnModel(nn.Module):
    def __init__(self,color_channels=1,units=1,dim=64,frames=8):
        super(cnnModel, self).__init__()

        FILM = torch.jit.load('./models/film_net_fp32.pt')
        FILM.eval()
        FILM.float()
        self.FILM = FILM.cuda()

            
    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,:,ba[0]::2,ba[1]::2]
        y_bayer = rearrange(y_bayer,"b f h w ba->(b ba) f h w")
        Phi_bayer = rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = rearrange(Phi_s_bayer,"b f h w ba->(b ba) f h w")

        x = At(y_bayer,Phi_bayer)
        yb = A(x,Phi_bayer)
        x = x + At(torch.div(y_bayer-yb,Phi_s_bayer),Phi_bayer)
        x = rearrange(x,"(b ba) f h w->b f h w ba",b=b)
        x_bayer = torch.zeros(b,f,h,w).to(y.device)
        for ib in range(len(bayer)): 
            ba = bayer[ib]
            x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
        x = x_bayer.unsqueeze(1)
        return x

    def forward(self, y, args):


        de_meas = inference(self.FILM,y[:,0:1,:,:].repeat(1,3,1,1),y[:,-1:,:,:].repeat(1,3,1,1),args.frames-2)
        de_meas = [torch.sum(de_meas[k],dim=1,keepdim=True).cuda() for k in range(args.frames)]

        return de_meas
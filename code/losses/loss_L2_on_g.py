import torch
import torch.nn as nn
import einops
from config.utils.utils import A, At
from config.utils.mask import generate_masks

class Loss(nn.Module):
    
    def __init__(self,cfg):
        super(Loss, self).__init__()
        mask,mask_s = generate_masks(cfg.train_data.mask_path,cfg.train_data.mask_shape)
        batch_size = cfg.data['samples_per_gpu']
        self.Phi = torch.from_numpy(einops.repeat(mask,'cr h w->b cr h w',b=batch_size)).cuda()
        self.Phi_s = torch.from_numpy(einops.repeat(mask_s,'h w->b 1 h w',b=batch_size)).cuda()

    def forward(self, f_hat, f_gt):
        g_gt = A(f_gt,self.Phi)
        g_hat = A(f_hat,self.Phi)
        L2 = torch.sqrt(torch.mean((g_hat-g_gt)**2))
        loss =  L2      
        return loss
import matplotlib.pyplot as plt
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from functions.utils import *
from functions.GetNetwork import *
from functions.adquisition import *
import tqdm
import torch
import torch.optim
import torch.nn as nn
from torchmetrics import TotalVariation


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--iter', default=3000, type=int, help='max epoch')
parser.add_argument('--LR', default=0.005, type=float)
parser.add_argument('--frames', default=16, type=int, help='compressive rate')
parser.add_argument('--size', default=[256, 340], type=int, help='input image resolution')
parser.add_argument('--input', default='./input/', type=str, help='input path')
parser.add_argument('--output', default='./output/', type=str, help='output path')
parser.add_argument('--name', default='snapshot.tiff', type=str, help='input path')
parser.add_argument('--network', default='3DUNet', type=str, help='input path')
parser.add_argument('--inputType', default='hybrid', type=str, help='input path')
parser.add_argument('--noiselvl', default=0.005, type=float)
parser.add_argument('--code', default=[1,0,1,1,1,0,0,0,1,0,1,1,0,1,1,1], type=int, help='Illumination Code')
args = parser.parse_args()


if not os.path.exists(args.output):
        os.makedirs(args.output)


#Define Net
net = getnetwork(args)
net = net.cuda()

#Define Loss Fn
loss_meas = nn.MSELoss()
loss_tv = TotalVariation()
loss_l1 = nn.L1Loss()

loss_meas  = loss_meas.cuda()
loss_tv    = loss_tv.cuda()
loss_l1 = loss_l1.cuda()

optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=args.LR)

gt,imgt = meas2tensor(args)
mask = code2tensor(args)

tensor_in = torch.unsqueeze(inputTensor(args),0)


# train loop
iter_loss = None
for it in range(args.iter):

   optimizer.zero_grad()
   
   if args.noiselvl !=0:
      tensor_input = tensor_in + torch.rand(size=tensor_in.shape,device=tensor_in.device)*args.noiselvl
   else:
      tensor_input = tensor_in
       
   
   datacube = net(tensor_input)
   
   meas = adquisition(datacube,mask,args)
   
   #Compute Losses
   lm = loss_meas(torch.squeeze(meas), torch.squeeze(gt))
   ltv = loss_tv(torch.squeeze(datacube))
   l1 = loss_l1(torch.squeeze(datacube))
   
   Loss = lm + 0.3*ltv + 0.3*l1

   Loss.backward()
   optimizer.step()
     
   if iter_loss is not None:
      iter_loss = np.concatenate((iter_loss,[Loss.detach().cpu().numpy()]),0)
   else:
      iter_loss =  [Loss.detach().cpu().numpy()]
   print("===> Iteration {} Complete: Avg. Loss: {:.12f}".format(it, Loss.detach().cpu().numpy()))

save2Mat(datacube,meas,gt,args)



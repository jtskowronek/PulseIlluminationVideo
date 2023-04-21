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


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--iter', default=8000, type=int, help='max epoch')
parser.add_argument('--LR', default=0.005, type=float, help='learning rate')
parser.add_argument('--alpha1', default=1e-6, type=float, help='weigth for TV')
parser.add_argument('--alpha2', default=1e-6, type=float, help='weigth for L1')
parser.add_argument('--saveEach', default=400, type=int, help='max epoch')
parser.add_argument('--frames', default=16, type=int, help='compressive rate')
parser.add_argument('--size', default=[256, 340], type=int, help='input image resolution')
parser.add_argument('--input', default='./input/', type=str, help='input path')
parser.add_argument('--output', default='./output/skip_a6b6/', type=str, help='output path')
parser.add_argument('--name', default='snapshot1.tiff', type=str, help='input path')
parser.add_argument('--nameRef', default='snapshot2.tiff', type=str, help='input path')
parser.add_argument('--network', default='skip', type=str, help='input path')
parser.add_argument('--inputType', default='noise', type=str, help='input path')
parser.add_argument('--noiselvl', default=0.001, type=float)
parser.add_argument('--code', default=[1,0,1,1,1,0,0,0,1,0,1,1,0,1,1,1], type=int, help='Illumination Code')
args = parser.parse_args()


if not os.path.exists(args.output):
        os.makedirs(args.output)


#Define Net
net = getnetwork(args)
net = net.cuda()

#Define Loss Fn
loss_meas = nn.MSELoss()
loss_tv = tv3_loss
loss_l1 = nn.L1Loss()

loss_meas  = loss_meas.cuda()

optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=args.LR)

gt,gtr,imgt,imr = meas2tensor(args)
mask = code2tensor(args)

tensor_in = inputTensor(args)
#tensor_in = torch.unsqueeze(inputTensor(args),0)


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
   ltv = loss_tv(datacube)
   l1 = torch.norm(torch.squeeze(datacube),1)
   
   Loss = lm + args.alpha1*ltv + args.alpha2*l1

   Loss.backward()
   optimizer.step()
     
   if iter_loss is not None:
      iter_loss = np.concatenate((iter_loss,[Loss.detach().cpu().numpy()]),0)
   else:
      iter_loss =  [Loss.detach().cpu().numpy()]
   
   
   if it % args.saveEach==0:
      save2Mat(datacube,meas,gt,iter_loss,args,it)
      
   if it == 1:
       meas_b = meas
       datacube_b = datacube
       loss_b = Loss.detach().cpu().numpy()
   
   if it > 1 and Loss.detach().cpu().numpy() < loss_b:
       meas_b = meas
       datacube_b = datacube
       loss_b = Loss.detach().cpu().numpy()
       save2Mat(datacube_b,meas_b,gt,iter_loss,args,"best")
       print("New Best!")
       


   print("===> Iter: %d - Total Loss: %.6f = %.2E + %.2E + %.2E" % (it,
       Loss.detach().cpu().numpy(),
       lm.detach().cpu().numpy(),
       args.alpha1*ltv.detach().cpu().numpy(),
       args.alpha2*l1.detach().cpu().numpy()
       ))    
    



save2Mat(datacube_b,meas_b,gt,iter_loss,args,"last")


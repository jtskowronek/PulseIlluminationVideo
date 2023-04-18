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

from skimage.measure import compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--iter', default=8000, type=int, help='max epoch')
parser.add_argument('--LR', default=0.01, type=float)
parser.add_argument('--frames', default=16, type=int, help='compressive rate')
parser.add_argument('--size', default=[256, 340], type=int, help='input image resolution')
parser.add_argument('--input', default='./input/', type=str, help='input path')
parser.add_argument('--output', default='./output/', type=str, help='output path')
parser.add_argument('--name', default='snapshot.tiff', type=str, help='input path')
parser.add_argument('--network', default='pat', type=str, help='input path')
parser.add_argument('--inputType', default='noise', type=str, help='input path')
parser.add_argument('--noiselvl', default=0.01, type=float)
parser.add_argument('--code', default=[1,0,1,1,1,0,0,0,1,0,1,1,0,1,1,1], type=int, help='Code')
args = parser.parse_args()


if not os.path.exists(args.output):
        os.makedirs(args.output)


#Define Net
net = getnetwork(args)
net = net.cuda()
loss = nn.MSELoss()
loss.cuda()

optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=args.LR)

gt,imgt = meas2tensor(args)
mask = code2tensor(args)

tensor_in = inputTensor(args)


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
   
   Loss = loss(torch.squeeze(meas), torch.squeeze(gt))
   Loss.backward()
   optimizer.step()
     
   if iter_loss is not None:
      iter_loss = np.concatenate((iter_loss,[Loss.detach().cpu().numpy()]),0)
   else:
      iter_loss =  [Loss.detach().cpu().numpy()]
   print("===> Iteration {} Complete: Avg. Loss: {:.12f}".format(it, Loss.detach().cpu().numpy()))

save2Mat(datacube,meas,gt,args)



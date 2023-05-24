import os
import os.path as osp
import sys 
BASE_DIR=osp.dirname(osp.dirname(__file__))
sys.path.append(BASE_DIR)
import torch 
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.utils import save_single_image,get_device_info,load_checkpoints
from cacti.utils.config import Config
from cacti.models.builder import build_model
from cacti.datasets.builder import build_dataset 
from cacti.utils.logger import Logger
from torch.cuda.amp import autocast
import numpy as np 
import argparse 
import einops 
import time 
from unet import UNet
from inverse import  StandardConv2D
from Exp_dataloader import Imgdataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from paralelize_patches import *
from scipy.io import savemat
import matplotlib.pyplot as plt

rsize = 128

parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,default='configs/STFormer/stformer_base.py')
parser.add_argument("--test_path",type=str,default="./test_datasets/meas3r16/test3/" + str(rsize) +"/")
parser.add_argument("--save_path",type=str,default='./test_datasets/meas3r16/test3/r' + str(rsize) +'/')
parser.add_argument("--demodel_path",type=str,default='')
parser.add_argument("--stt_path",type=str,default='./train_results/full_model/checkpoints/epoch_156.pth')
parser.add_argument("--device",type=str,default="cuda")
parser.add_argument('--size', default=[rsize,rsize], type=int, help='input image resolution')
parser.add_argument('--frames', default=14, type=int, help='input image frames')
parser.add_argument("--local_rank",default=-1)
args = parser.parse_args()


if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

cfg = Config.fromfile(args.config)
model = build_model(cfg.model).to(args.device)  
DeModel = UNet(in_channel=16, out_channel=14, instance_norm=False).to(args.device)  


resume_dict = torch.load(args.stt_path)
model_state_dict = resume_dict["Demodel_state_dict"]
load_checkpoints(DeModel,model_state_dict)

for name, para in DeModel.named_parameters():
   para.requires_grad = False
    
   
resume_dict = torch.load(args.stt_path)
model_state_dict = resume_dict["model_state_dict"]
load_checkpoints(model,model_state_dict)

for name, para in model.named_parameters():
   para.requires_grad = False

mask,mask_s = generate_masks(cfg.train_data.mask_path,cfg.train_data.mask_shape)
dataset2 = Imgdataset(args.test_path)
test_data_loader = DataLoader(dataset=dataset2, batch_size=1, shuffle=False)

Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)
Phi = torch.from_numpy(Phi).to(args.device)
Phi_s = torch.from_numpy(Phi_s).to(args.device)    

kernel_size = 128
stride = kernel_size//4

dummy = extract_patches_2d(torch.zeros(size=(1,1,args.size[0],args.size[0])), kernel_size=kernel_size, stride=stride)
    
npatch = dummy.shape[0]
nmeas = 3

def plot(T):
    
    torch.squeeze(T)
    T = T.cpu().numpy()
    plt.imshow(T)
    plt.show()


for iter,data in enumerate(test_data_loader):
    fname = test_data_loader.dataset.data[iter]['data'][-8:-4]
    print("Reconstruction: {}".format(fname))
    meas1 = data[0].float().to(args.device)    
    measc = data[1].float().to(args.device)
    meas2 = data[2].float().to(args.device)  
    
    meas_f = torch.cat((meas1[:,0:1,:,:],measc,meas2[:,-1:,:,:]),1)
    
    
    patches = torch.zeros(size=(npatch,nmeas,kernel_size,kernel_size),device=args.device)      
    for ptc in range(nmeas):
       patches[:,ptc:ptc+1,:,:] = extract_patches_2d(meas_f[:,ptc:ptc+1,:,:], kernel_size=kernel_size, stride=stride)
    
    out = torch.zeros(size=(patches.shape[0],args.frames,128,128),device=args.device)
    full = torch.zeros(size=(1,args.frames,args.size[0],args.size[1]),device=args.device)
    for ph in range(patches.shape[0]):   
        de_meas = DeModel(patches[ph:ph+1,:,:,:],Phi,Phi_s)
        out_s = model(de_meas,Phi,Phi_s)
        out[ph:ph+1,:,:,:] = out_s[0]
    
    #out = torch.cat((meas1[:,0:1,:,:],out[:,1:-1,:,:],meas2[:,-1:,:,:]),1)    
    
    for f in range(args.frames) : 
        
        full_s = combine_patches_2d(out[:,f:f+1,:,:], kernel_size=kernel_size, output_shape=meas1.shape, stride=stride)
        fnorm = torch.ones(size=out[:,f:f+1,:,:].shape, device=args.device)
        fnorm = combine_patches_2d(fnorm, kernel_size=kernel_size, output_shape=meas1.shape, stride=stride)
        
        full[:,f:f+1,:,:] = full_s/fnorm
     
    full = torch.cat((meas1[:,0:1,:,:],full,meas2[:,-1:,:,:]),1)  
    full_out = torch.permute(full[0,:,:,:],(1,2,0)).cpu().numpy()
    meas_out = torch.squeeze(meas_f).cpu().numpy()
    sname = "recon_" + fname + ".mat"    
    savemat(args.save_path+sname, {'recon': full_out,'meas':meas_out})    
        




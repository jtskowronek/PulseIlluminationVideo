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

parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,default='configs/STFormer/stformer_base.py')
parser.add_argument("--test_path",type=str,default='./experimental_data/cube6/512/in/')
parser.add_argument("--save_path",type=str,default='./experimental_data/cube6/512/recon/')
parser.add_argument("--demodel_path",type=str,default='./coarse_net/checkpoints/epoch_398.pth')
parser.add_argument("--stt_path",type=str,default='./pulsed_check/checkpoints/epoch_22.pth')
parser.add_argument("--device",type=str,default="cuda")
parser.add_argument('--size', default=[512, 512], type=int, help='input image resolution')
parser.add_argument("--local_rank",default=-1)
args = parser.parse_args()


if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

cfg = Config.fromfile(args.config)
model = build_model(cfg.model).to(args.device)  
DeModel = UNet(in_channel=64, out_channel=8, instance_norm=False).to(args.device)  


resume_dict = torch.load(args.demodel_path)
model_state_dict = resume_dict["model_state_dict"]
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
stride = 128



for iter,data in enumerate(test_data_loader):
    fname = test_data_loader.dataset.data[iter]['data'][-8:-4]
    print("Reconstruction: {}".format(fname))
    meas = data.float().to(args.device)    
    patches = extract_patches_2d(meas, kernel_size=kernel_size, stride=stride)
    
    out = torch.zeros(size=(patches.shape[0],8,128,128),device=args.device)
    full = torch.zeros(size=(1,8,args.size[0],args.size[1]),device=args.device)
    for p in range(patches.shape[0]):   
        de_meas = DeModel(patches[p:p+1,:,:,:])
        out_s = model(de_meas,Phi,Phi_s)
        out[p:p+1,:,:,:] = out_s[0]
        
    for f in range(8) : 
        
        full_s = combine_patches_2d(out[:,f:f+1,:,:], kernel_size=kernel_size, output_shape=meas.shape, stride=stride)
        fnorm = torch.ones(size=out[:,f:f+1,:,:].shape, device=args.device)
        fnorm = combine_patches_2d(fnorm, kernel_size=kernel_size, output_shape=meas.shape, stride=stride)
        
        full[:,f:f+1,:,:] = full_s/fnorm
     
    full_out = torch.permute(full[0,:,:,:],(1,2,0)).cpu().numpy()
    meas_out = torch.squeeze(meas).cpu().numpy()
    sname = "recon_" + fname + ".mat"    
    savemat(args.save_path+sname, {'recon': full_out,'meas':meas_out})    
        




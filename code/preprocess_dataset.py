import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
from torch.utils.data import DataLoader
from config.dataset.builder import build_dataset
from config.utils.mask import generate_masks
from config.utils.utils import get_device_info
from config.util_functions import *
import scipy.io as sio
import torch
import time
import argparse 
from tqdm import tqdm
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,default='./config/default_config.py')
parser.add_argument("--train_dataset_path",type=str,default='./dataset/DAVIS/JPEGImages/480p/')
parser.add_argument("--preprocess_path_train",type=str,default='./dataset/Preprocess_DAVIS/')
parser.add_argument("--preprocess_path_test", type=str,default='./dataset/Preprocess_test_dataset/')
parser.add_argument("--mask_path",type=str,default='./masks/shutter_mask16.mat')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument("--resolution",type=eval,default=[128,128])
parser.add_argument("--frames",type=int,default=16)
parser.add_argument("--dataset_crop",type=eval,default=[128,128])
parser.add_argument("--batch_size",type=int,default=1)
args = parser.parse_args()
args.device = "cuda"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {} using {}'.format(n_gpu,args.gpu))


if __name__ == '__main__':

    cfg = Config.fromfile(args.config)
    cfg.resize_h,cfg.resize_w = args.resolution
    cfg.crop_h,cfg.crop_w = args.dataset_crop
    
    cfg.train_pipeline[4]['resize_h'],cfg.train_pipeline[4]['resize_w'] = args.resolution
    cfg.train_pipeline[1]['crop_h'],cfg.train_pipeline[1]['crop_w'] = args.dataset_crop
    cfg.train_data['data_root'] = args.train_dataset_path
    cfg.train_data.mask_path = args.mask_path
    cfg.train_data.mask_shape = (args.resolution[0],args.resolution[1],args.frames)
    cfg.data['samples_per_gpu'] = args.batch_size
    

    save_path_train = args.preprocess_path_train
    save_path_test = args.preprocess_path_test

    if not osp.exists(save_path_train):
        os.makedirs(save_path_train)
    if not osp.exists(save_path_test):
        os.makedirs(save_path_test)

    device_info = get_device_info()

    ## Load cuda devide
    device = args.device




    ## Preparing dataset
    mask,mask_s = generate_masks(cfg.train_data.mask_path,cfg.train_data.mask_shape)
    train_data = build_dataset(cfg.train_data,{"mask":mask})
    train_data_loader = DataLoader(dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        shuffle=True,
                                        num_workers = cfg.data.workers_per_gpu)
    
    test_data_loader = DataLoader(dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        shuffle=True)
    

    iter_num = len(train_data_loader) 
    start_time = time.time()
    for iteration, data in tqdm(enumerate(train_data_loader),
                                 desc ="Processing training data... ",colour="red",
                                 total=iter_num,
                                 ascii=' 123456789═'):
        gt, meas = data
        gt = gt[0].float()
        meas = meas.float()
        torch.save([meas,gt],save_path_train+f"data_{iteration}.pth")
    end_time = time.time()


################



    
    iter_num = len(test_data_loader) 
    start_time = time.time()
    for iteration, data in tqdm(enumerate(test_data_loader),
                                 desc ="Processing testing data... ",colour="green",
                                 total=30,
                                 ascii=' 123456789═'):
        
        gt, meas = data
        gt = gt[0].float().to(args.device)
        meas = meas.float().to(args.device)
        torch.save([meas,gt],save_path_test+f"data_{iteration}.pth")
        if iteration > 30:
           break
    end_time = time.time()        
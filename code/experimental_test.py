import os
import os.path as osp
import sys 
import importlib
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
from torch.utils.data import DataLoader
from config.dataset.builder import build_dataset
from config.utils.mask import generate_masks
from config.utils.logger import Logger
from config.utils.utils import save_image, load_checkpoints, get_device_info
from config.utils.eval_full import eval_psnr_ssim
from config.util_functions import *
from config.experimental_loader import *
import torch
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,default='./config/default_config.py')
parser.add_argument("--work_dir",type=str,default='/full_modelv4/')
parser.add_argument("--test_dataset_path",type=str,default='./dataset/test_dataset/')
parser.add_argument("--mask_path",type=str,default='./masks/shutter_mask16.mat')
parser.add_argument("--model_module",type=str,default='base_model')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument("--resolution",type=eval,default=[64,64])
parser.add_argument("--frames",type=int,default=16)
parser.add_argument("--checkpoints",type=str,default=None)

args = parser.parse_args()
args.device = "cuda"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {} using {}'.format(n_gpu,args.gpu))


if __name__ == '__main__':


    ## Load cuda devide
    device = args.device

    ## Import model
    CNNmethod = importlib.import_module('models.'+ args.model_module)
    model = CNNmethod.cnnModel(frames=args.frames).to(args.device)
    model = model.eval()

    test_dataset = Imgdataset(args.test_dataset_path)

    for iteration, data in enumerate(test_dataset):
        meas0,meas1,measc = data
        meas0 = meas0.to(args.device)
        meas1 = meas1.to(args.device)
        measc = measc.to(args.device)

        meas_f = torch.cat((meas0,measc,meas1),1)

        model_out = model(meas_f,args)
        model_out_f = torch.cat((meas0,model_out[:,1:-1,:,:],meas1),1)
        



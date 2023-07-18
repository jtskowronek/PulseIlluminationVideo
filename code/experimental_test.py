import os
import os.path as osp
import sys 
import importlib
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
from config.dataset.builder import build_dataset
from config.utils.mask import generate_masks
from config.utils.logger import Logger
from config.utils.utils import save_image, load_checkpoints, get_device_info
from config.utils.eval_full import eval_psnr_ssim
from config.util_functions import *
from config.experimental_loader import *
import torch
from tqdm import tqdm
import argparse 
from torch.utils.data import DataLoader
import scipy as sci


parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,default='./config/default_config.py')
parser.add_argument("--work_dir",type=str,default='/full_modelv4/')
parser.add_argument("--test_dataset_path",type=str,default='./dataset/processed/coded/test6/256/')
parser.add_argument("--results_path",type=str,default='./dataset/processed/coded/test6/256r/')
parser.add_argument("--mask_path",type=str,default='./masks/shutter_mask16.mat')
parser.add_argument("--model_module",type=str,default='base_model')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument("--resolution",type=eval,default=[128,128])
parser.add_argument("--frames",type=int,default=16)
parser.add_argument("--sub_sampling",type=int,default=1)
parser.add_argument("--checkpoints",type=str,default='./training_results/baseline/checkpoint/epoch_127.pth')

args = parser.parse_args()
args.device = "cuda"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {} using {}'.format(n_gpu,args.gpu))


if __name__ == '__main__':


    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    ## Load cuda devide
    device = args.device

    ## Import model
    CNNmethod = importlib.import_module('models.'+ args.model_module)
    model = CNNmethod.cnnModel(frames=args.frames).to(args.device)
    model = model.eval()
    resume_dict = torch.load(args.checkpoints)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict)

    test_dataset = Imgdataset(args.test_dataset_path,)
    experimental_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    for iteration, data in tqdm(enumerate(experimental_dataloader),
                                 desc ="Reconstructing... ",colour="red",
                                 total=len(experimental_dataloader),
                                 ascii=' 123456789═'):
        meas1,meas2,measc,name = data
        meas1 = meas1.to(args.device).float()
        meas2 = meas2.to(args.device).float()
        measc = measc.to(args.device).float()

        meas_f = torch.cat((meas1,measc,meas2),1)


        with torch.no_grad():
            model_out = model(meas_f,args) 


        model_out_f = torch.cat((meas1,model_out[:,1:-1,:,:],meas2),1)

        out_save = torch.squeeze(model_out_f).permute(1,2,0).cpu().numpy()

        scio.savemat(args.results_path + name[0], {'recon': out_save})
        



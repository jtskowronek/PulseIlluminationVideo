import os
import os.path as osp
import sys 
import importlib
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import bisect
from torch.utils.data import DataLoader

from config.dataset.builder import build_dataset
from config.utils.mask import generate_masks
from config.config_file import Config
from config.utils.logger import Logger
from config.utils.utils import save_image, load_checkpoints, get_device_info
from config.utils.eval_full import eval_psnr_ssim
from config.util_functions import *
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import argparse 
import json 
import einops


parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,default='./config/default_config.py')
parser.add_argument("--work_dir",type=str,default='/full_modelv4/')
parser.add_argument("--train_dataset_path",type=str,default='./dataset/DAVIS/JPEGImages/480p/')
parser.add_argument("--test_dataset_path",type=str,default='./dataset/test_dataset/')
parser.add_argument("--mask_path",type=str,default='./masks/shutter_mask16.mat')
parser.add_argument("--model_module",type=str,default='base_model')
parser.add_argument("--loss_module",type=str,default='loss_test2')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument("--resolution",type=int,default=[64,64])
parser.add_argument("--frames",type=int,default=16)
parser.add_argument("--dataset_crop",type=int,default=[32,32])
parser.add_argument("--resume",type=str,default=None)
parser.add_argument("--Epochs",type=int,default=400)
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--learning_rate",type=int,default=0.0001)
parser.add_argument("--saveImageEach",type=int,default=500)
parser.add_argument("--saveModelEach",type=int,default=1)
parser.add_argument("--checkpoints",type=str,default=None)
args = parser.parse_args()
args.device = "cuda"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {} using {}'.format(n_gpu,args.gpu))


if __name__ == '__main__':

    cfg = update_cfg(args)
    
    logger = Logger(cfg.log_dir)
    writer = SummaryWriter(log_dir = cfg.show_dir)

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])

    ## Load cuda devide
    device = args.device

    ## Import model
    CNNmethod = importlib.import_module('models.'+ args.model_module)
    model = CNNmethod.cnnModel(frames=args.frames).to(args.device)

    ## Import Loss()
    LossMethod = importlib.import_module('losses.'+ args.loss_module)
    loss_function = LossMethod.Loss(cfg).to(args.device)
    
    ## Optimizer
    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=args.learning_rate)

    logger.info('GPU info:\n' 
                + dash_line + 
                env_info + '\n' +
                dash_line)
    logger.info('cfg info:\n'
                + dash_line + 
                json.dumps(cfg, indent=4)+'\n'+
                dash_line) 
    logger.info('Model info:\n'
                + dash_line + 
                str(model)+'\n'+
                dash_line)


    ## Preparing dataset
    mask,mask_s = generate_masks(cfg.train_data.mask_path,cfg.train_data.mask_shape)
    train_data = build_dataset(cfg.train_data,{"mask":mask})
    test_data = build_dataset(cfg.test_data,{"mask":mask})
    train_data_loader = DataLoader(dataset=train_data, 
                                        batch_size=cfg.data.samples_per_gpu,
                                        shuffle=True,
                                        num_workers = cfg.data.workers_per_gpu)
    
    
    


    
    
    start_epoch = 0
    if cfg.checkpoints is not None:
            logger.info("Load pre_train model...")
            resume_dict = torch.load(cfg.checkpoints)
            if "model_state_dict" not in resume_dict.keys():
                model_state_dict = resume_dict
            else:
                model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)
    else:            
            logger.info("No pre_train model")

    if cfg.resume is not None:
            logger.info("Load resume...")
            resume_dict = torch.load(cfg.resume)
            start_epoch = resume_dict["epoch"]
            model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)

            optim_state_dict = resume_dict["optim_state_dict"]
            optimizer.load_state_dict(optim_state_dict)


    iter_num = len(train_data_loader) 
    for epoch in range(start_epoch,cfg.runner.max_epochs):
        epoch_loss = 0
        model = model.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            gt, meas = data
            gt = gt.float().to(args.device)
            meas = meas.unsqueeze(1).float().to(args.device)
            batch_size = meas.shape[0]

            #Phi = einops.repeat(mask,'cr h w->b cr h w',b=batch_size)
            #Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=batch_size)

            #Phi = torch.from_numpy(Phi).to(args.device)
            #Phi_s = torch.from_numpy(Phi_s).to(args.device)

            optimizer.zero_grad()
            meas_f = torch.cat((gt[:,0:1,:,:],meas,gt[:,-1:,:,:]),1)

            model_out = model(meas_f,args)
            model_out_f = torch.cat((gt[:,0:1,:,:],model_out[:,1:-1,:,:],gt[:,-1:,:,:]),1)
            

            if not isinstance(model_out,list):
                model_out = [model_out_f]
            loss = loss_function(model_out_f, gt)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()


            if (iteration % cfg.log_config.interval) == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                iter_len = len(str(iter_num))
                logger.info("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}.".format(epoch,iteration,iter_len,iter_num,lr,loss.item()))
                writer.add_scalar("loss",loss.item(),epoch*len(train_data_loader) + iteration)
            if (iteration % cfg.save_image_config.interval) == 0:
                sing_out = model_out_f[0].detach().cpu().numpy()
                sing_gt = gt[0].cpu().numpy()
                sing_mask = mask
                image_name = osp.join(cfg.train_image_save_dir,str(epoch)+"_"+str(iteration)+".png")
                save_image(sing_out,sing_gt,sing_mask,image_name)
        end_time = time.time()

        logger.info("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.\n".format(epoch,epoch_loss/(iteration+1),end_time-start_time))

        if (epoch % cfg.checkpoint_config.interval) == 0:
 
            save_model = model
            checkpoint_dict = {
                "epoch": epoch, 
                "model_state_dict": save_model.state_dict(), 
                "optim_state_dict": optimizer.state_dict(), 
            }
            torch.save(checkpoint_dict,osp.join(cfg.checkpoints_dir,"epoch_"+str(epoch)+".pth")) 

        if cfg.eval.flag and epoch % cfg.eval.interval==0:
            psnr_dict,ssim_dict = eval_psnr_ssim(model,test_data,mask,mask_s,args)
            psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])           
            ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
            logger.info("Mean PSNR: \n{}.\n".format(psnr_str))
            logger.info("Mean SSIM: \n{}.\n".format(ssim_str))
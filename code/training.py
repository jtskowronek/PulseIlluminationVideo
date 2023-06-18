import os
import os.path as osp
import sys 
import importlib
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
from torch.utils.data import DataLoader
from config.dataset.builder import build_dataset
from config.utils.mask import generate_masks
from config.utils.utils import save_image, load_checkpoints, get_device_info
from config.utils.eval_full import eval_psnr_ssim
from config.util_functions import *
from config.setupFolders import setupFolders
from config.customDataloader import Imgdataset
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import argparse 
import json 


parser = argparse.ArgumentParser()
parser.add_argument("--experimentName",type=str,default='test')
parser.add_argument("--train_dataset_path",type=str,default='./dataset/Preprocess_DAVIS/')
parser.add_argument("--test_dataset_path",type=str,default='./dataset/Preprocess_test_dataset/')
parser.add_argument("--mask_path",type=str,default='./masks/shutter_mask16.mat')
parser.add_argument("--model_module",type=str,default='base_model')
parser.add_argument("--loss_module",type=str,default='loss_test2')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument("--resolution",type=eval,default=[64,64])
parser.add_argument("--frames",type=int,default=16)
parser.add_argument("--resume",type=str,default=None)
parser.add_argument("--checkpoints",type=str,default=None)
parser.add_argument("--Epochs",type=int,default=100)
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--learning_rate",type=int,default=0.0001)
parser.add_argument("--saveImageEach",type=int,default=500)
parser.add_argument("--saveModelEach",type=int,default=1)

args = parser.parse_args()
args.device = "cuda"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
n_gpu = torch.cuda.device_count()
print(torch.cuda.is_available())
print('The number of GPU is {} using {}'.format(n_gpu,args.gpu))


if __name__ == '__main__':

    ## Setup folders
    paths = setupFolders(args)
    writer = SummaryWriter(log_dir = paths.tb_path)

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    args.vid_shape = [args.resolution[0],args.resolution[1],args.frames]
    ## Load cuda devide
    device = args.device

    ## Import model
    CNNmethod = importlib.import_module('models.'+ args.model_module)
    model = CNNmethod.cnnModel(frames=args.frames).to(args.device)

    ## Import Loss()
    LossMethod = importlib.import_module('losses.'+ args.loss_module)
    loss_function = LossMethod.Loss(args).to(args.device)
    
    ## Init optimizer
    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=args.learning_rate)

    ## Preparing dataset
    mask,mask_s = generate_masks(args.mask_path,args.vid_shape)
    train_dataset = Imgdataset(args.train_dataset_path)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Imgdataset(args.test_dataset_path)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)  
    
    start_epoch = 0
    if args.checkpoints is not None:
            print("Load pre_train model...")
            resume_dict = torch.load(args.checkpoints)
            if "model_state_dict" not in resume_dict.keys():
                model_state_dict = resume_dict
            else:
                model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)
    else:            
            print("No pre_train model")

    if args.resume is not None:
            print("Load resume...")
            resume_dict = torch.load(args.resume)
            start_epoch = resume_dict["epoch"]
            model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model,model_state_dict)

            optim_state_dict = resume_dict["optim_state_dict"]
            optimizer.load_state_dict(optim_state_dict)


    iter_num = len(train_data_loader) 
    for epoch in range(start_epoch,args.Epochs):
        epoch_loss = 0
        optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=args.learning_rate)
        model = model.train()
        start_time = time.time()
        for iteration, data in tqdm(enumerate(train_data_loader),
                                 desc ="Training... ",colour="red",
                                 total=iter_num,
                                 ascii=' 123456789═'):
            gt, meas = data
            gt = gt.to(args.device)
            meas = meas.to(args.device)

            optimizer.zero_grad()
            meas_f = torch.cat((gt[:,0:1,:,:],meas,gt[:,-1:,:,:]),1)

            model_out = model(meas_f,args)
            model_out_f = torch.cat((gt[:,0:1,:,:],model_out[:,1:-1,:,:],gt[:,-1:,:,:]),1)
            

            loss = loss_function(model_out_f, gt)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()


            if (iteration % 100) == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                iter_len = len(str(iter_num))
                print("epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}.".format(epoch,iteration,iter_len,iter_num,lr,loss.item()))
                writer.add_scalar("loss",loss.item(),epoch*len(train_data_loader) + iteration)
                
            if (iteration % args.saveImageEach) == 0:
                sing_out = model_out_f[0].detach().cpu().numpy()
                sing_gt = gt[0].cpu().numpy()
                sing_mask = mask
                image_name = osp.join(paths.result_path,str(epoch)+"_"+str(iteration)+".png")
                save_image(sing_out,sing_gt,sing_mask,image_name)
        end_time = time.time()

        

        print("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.\n".format(epoch,epoch_loss/(iteration+1),end_time-start_time))

        if (epoch % args.saveModelEach) == 0:
 
            save_model = model
            checkpoint_dict = {
                "epoch": epoch, 
                "model_state_dict": save_model.state_dict(), 
                "optim_state_dict": optimizer.state_dict(), 
            }
            torch.save(checkpoint_dict,osp.join(paths.trained_model_path,"epoch_"+str(epoch)+".pth")) 

        psnr_dict,ssim_dict = eval_psnr_ssim(model.eval(),test_data_loader,mask,args,paths)
        psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])           
        ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
        psnr_values = list(psnr_dict.values())
        ssim_values = list(ssim_dict.values())
        print("Mean PSNR: \n{}.\n".format(psnr_str))
        print("Mean SSIM: \n{}.\n".format(ssim_str))
        writer.add_scalar("psnr_metric",psnr_values[-1],epoch*len(train_data_loader) + iteration)
        writer.add_scalar("ssim_metric",ssim_values[-1],epoch*len(train_data_loader) + iteration)
        if (epoch % 5 == 0) and (epoch < 100):
            args.learning_rate = args.learning_rate * 0.95  


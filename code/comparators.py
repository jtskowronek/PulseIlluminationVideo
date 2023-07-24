import os
import os.path as osp
import sys 
import importlib
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
from config.utils.utils import load_checkpoints, normalize_tensor
from config.util_functions import *
from config.customDataloader import *
import torch
from tqdm import tqdm
import argparse 
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,default='./config/default_config.py')
parser.add_argument("--work_dir",type=str,default='/full_modelv4/')
parser.add_argument("--test_dataset_path",type=str,default='./dataset/Preprocess_test_dataset/')
parser.add_argument("--mask_path",type=str,default='./masks/shutter_mask16.mat')
parser.add_argument("--model_module",type=str,default='base_model')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument("--resolution",type=eval,default=[128,128])
parser.add_argument("--frames",type=int,default=16)
parser.add_argument("--normalize",type=int,default=0)
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

    ## Load cuda devide
    device = args.device

    ## Import models
    CNNmethod = importlib.import_module('models.'+ args.model_module)
    model = CNNmethod.cnnModel(frames=args.frames).to(args.device)
    model = model.eval()
    resume_dict = torch.load(args.checkpoints)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict)

    CNNmethod = importlib.import_module('models.FILM')
    modelFILM = CNNmethod.cnnModel(frames=args.frames).to(args.device)
    modelFILM = modelFILM.eval()

    CNNmethod = importlib.import_module('models.EDSC_f')
    modelEDSC = CNNmethod.cnnModel(frames=args.frames).to(args.device)
    modelEDSC = modelEDSC.eval()

    test_dataset = Imgdataset(args.test_dataset_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    ssim = StructuralSimilarityIndexMeasure(data_range=255)
    psnr = PeakSignalNoiseRatio(data_range=255)
    PSNRV,SSIMV,TIMEV = [],[],[]
    PSNRF,SSIMF,TIMEF = [],[],[]
    PSNRE,SSIME,TIMEE = [],[],[]
    for iteration, data in tqdm(enumerate(test_dataloader),
                                 desc ="Reconstructing... ",colour="red",
                                 total=len(test_dataloader),
                                 ascii=' 123456789═'):
        gt, meas = data
        gt = gt.to(args.device)
        meas = meas.to(args.device)

        meas_f = torch.cat((gt[:,0:1,:,:],meas,gt[:,-1:,:,:]),1)
        ref_f  = torch.cat((gt[:,0:1,:,:],gt[:,6:7,:,:],gt[:,-1:,:,:]),1)

        with torch.no_grad():

            t1 = time.time()
            model_out = model(meas_f,args) 
            t_model = time.time()-t1

            t2 = time.time()
            ref_out   = modelFILM(ref_f,args)
            t_film = time.time()-t2

            t3 = time.time()
            EDS_out   = modelEDSC(ref_f,args)
            t_edsc = time.time()-t3


        if args.normalize == 1:
        
            gtc = normalize_tensor(gt).cpu()
            moc = normalize_tensor(model_out).cpu()
            ref = normalize_tensor(ref_out).cpu()
            ref2 = normalize_tensor(EDS_out).cpu()
        else:
            gtc = gt.cpu()
            moc = model_out.cpu()
            ref = ref_out.cpu()
            ref2 = EDS_out.cpu()

        psnr_val = psnr(gtc*255,moc*255)
        ssim_val = ssim(gtc*255,moc*255) 

        PSNRV.append(psnr_val) 
        SSIMV.append(ssim_val)
        TIMEV.append(t_model)     

        psnr_ref = psnr(normalize_tensor(gtc)*255,normalize_tensor(ref)*255)
        ssim_ref = ssim(normalize_tensor(gtc)*255,normalize_tensor(ref)*255)

        PSNRF.append(psnr_ref) 
        SSIMF.append(ssim_ref)  
        TIMEF.append(t_film) 

        psnr_eds = psnr(normalize_tensor(gtc)*255,normalize_tensor(ref2)*255)
        ssim_eds = ssim(normalize_tensor(gtc)*255,normalize_tensor(ref2)*255)

        PSNRE.append(psnr_eds) 
        SSIME.append(ssim_eds) 
        TIMEE.append(t_edsc)            

    PSNRV = torch.stack(PSNRV)
    SSIMV = torch.stack(SSIMV)

    PSNRF = torch.stack(PSNRF)
    SSIMF = torch.stack(SSIMF)

    PSNRE = torch.stack(PSNRE)
    SSIME = torch.stack(SSIME)



    mp_our = torch.mean(PSNRV).cpu().numpy()
    sp_our = torch.std(PSNRV).cpu().numpy()
    ms_our = torch.mean(SSIMV).cpu().numpy()
    ss_our = torch.std(SSIMV).cpu().numpy()   
    ti_our = np.mean(TIMEV)

    mp_fil = torch.mean(PSNRF).cpu().numpy()
    sp_fil = torch.std(PSNRF).cpu().numpy()
    ms_fil = torch.mean(SSIMF).cpu().numpy()
    ss_fil = torch.std(SSIMF).cpu().numpy()
    ti_fil = np.mean(TIMEF) 

    mp_eds = torch.mean(PSNRE).cpu().numpy()
    sp_eds = torch.std(PSNRE).cpu().numpy()
    ms_eds = torch.mean(SSIME).cpu().numpy()
    ss_eds = torch.std(SSIME).cpu().numpy() 
    ti_eds = np.mean(TIMEE)

    print(f"|   Method   |     PSNR      |      SSIM      | Inferece Time |")      
    print(f"|   Ours     | {mp_our:.2f} +- {sp_our:.2f} |  {ms_our:.4f} +- {ss_our:.4f}  | {ti_our:.2f} |") 
    print(f"|   FILM     | {mp_fil:.2f} +- {sp_fil:.2f} |  {ms_fil:.4f} +- {ss_fil:.4f}  | {ti_fil:.2f} |")
    print(f"|   EDSC     | {mp_eds:.2f} +- {sp_eds:.2f} |  {ms_eds:.4f} +- {ss_eds:.4f}  | {ti_eds:.2f} |") 

        



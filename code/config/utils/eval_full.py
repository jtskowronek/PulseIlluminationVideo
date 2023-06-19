import os
import os.path as osp
from torch.utils.data.dataloader import DataLoader 
import torch 
from config.utils.utils import save_image
from config.utils.metrics import compare_psnr,compare_ssim
import numpy as np 
import einops 
from tqdm import tqdm
import bisect



def eval_psnr_ssim(model,test_data,mask,args,paths):
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    cr = args.frames
    for iter,data in tqdm(enumerate(test_data),
                                 desc ="Validation... ",colour="green",
                                 total=len(test_data),
                                 ascii=' 123456789═'):
        psnr,ssim = 0,0
        batch_output = []

        gt, meas = data
        meas1 = gt[:,0:1].to(args.device)
        meas2 = gt[:,-1:].to(args.device)
        gt = gt[0].cpu().numpy()
        
        meas = meas.to(args.device)
        batch_size = 1
        
        meas_f = torch.cat((meas1,meas,meas2),1)
        with torch.no_grad():
            
            outputs = model(meas_f,args)
        if not isinstance(outputs,list):
            outputs = [outputs]
            
        
            
        model_out_f = torch.cat((meas1,outputs[0][:,1:-1,:,:],meas2),1)
        output = model_out_f[0].cpu().numpy()
        
        batch_output.append(output)
        for jj in range(cr):
            if output.shape[0]==3:
                per_frame_out = output[:,jj]
                per_frame_out = np.sum(per_frame_out*test_data.rgb2raw,axis=0)
            else:
                per_frame_out = output[jj]
            per_frame_gt = gt[jj, :, :]
            psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
            ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
        psnr = psnr / (batch_size * cr)
        ssim = ssim / (batch_size * cr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    test_dir = paths.result_path
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    for i,name in enumerate(test_data):
        _name = f"data_{i}"
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        image_name = osp.join(test_dir,_name+".png")
        save_image(out[0],gt,mask,image_name)
    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    return psnr_dict,ssim_dict
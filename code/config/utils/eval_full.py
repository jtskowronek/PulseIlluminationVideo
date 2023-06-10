import os
import os.path as osp
from torch.utils.data.dataloader import DataLoader 
import torch 
from config.utils.utils import save_image
from config.utils.metrics import compare_psnr,compare_ssim
import numpy as np 
import einops 
import bisect


def inference(Demodel, img_batch_1, img_batch_2, inter_frames):
        results = [
            img_batch_1,
            img_batch_2
        ]

        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)

        for _ in range(len(remains)):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            x0 = x0.cuda()
            x1 = x1.cuda()

            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = Demodel(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]
        return results


def eval_psnr_ssim(model,Demodel,test_data,mask,mask_s,args):
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    data_loader = DataLoader(test_data,1,shuffle=False,num_workers=4)
    cr = mask.shape[0]
    for iter,data in enumerate(data_loader):
        psnr,ssim = 0,0
        batch_output = []

        meas, gt = data
        meas1 = gt[0,:,0:1,:,:].float().to(args.device)
        meas2 = gt[0,:,-1:,:,:].float().to(args.device)
        gt = gt[0].numpy()
        
        meas = meas[0].float().to(args.device)
        batch_size = 1
         
        Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
        Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)

        Phi = torch.from_numpy(Phi).to(args.device)
        Phi_s = torch.from_numpy(Phi_s).to(args.device)
        
        for ii in range(batch_size):
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            meas_f = torch.cat((meas1,single_meas,meas2),1)
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
                per_frame_gt = gt[ii,jj, :, :]
                psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)
        psnr = psnr / (batch_size * cr)
        ssim = ssim / (batch_size * cr)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        out_list.append(np.array(batch_output))
        gt_list.append(gt)

    test_dir = osp.join(args.work_dir,"test_images")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    for i,name in enumerate(test_data.data_name_list):
        _name,_ = name.split("_")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        for j in range(out.shape[0]):
            image_name = osp.join(test_dir,_name+"_"+str(j)+".png")
            save_image(out[j],gt[j],mask,image_name)
    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    return psnr_dict,ssim_dict
import torch


def adquisition(vid,mask,params):
    
    meas = torch.sum(vid*mask,1)

    return meas
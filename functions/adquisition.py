import torchvision.transforms.functional as F
import torch


def adquisition(input,mask,params):
    meas =  F.resize(input,(params.size[0],params.size[1]))
    meas = torch.sum(meas*mask,1)

    return meas
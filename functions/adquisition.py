import torch


def adquisition(vid,mask):
    mask = torch.squeeze(mask)
    vid = torch.squeeze(vid)
    meas = torch.sum(vid*mask,0)

    return meas
import torch
import numpy as np

def code2tensor(params):

    code = np.array(params.code)
    code = torch.from_numpy(code)

    mask = torch.ones(size=(params.size[0],params.size[1],params.frames))
    mask = mask*code
    mask = torch.permute(mask,(2,0,1))
    mask = torch.unsqueeze(mask,0).cuda().float()
    return(mask)

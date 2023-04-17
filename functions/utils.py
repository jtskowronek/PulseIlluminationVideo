import torch
from PIL import Image
import numpy as np




def meas2tensor(params):
     fullpath = params.input+params.name
     im = Image.open(fullpath)
     input = np.array(im)
     input = torch.from_numpy(input)
     input = torch.unsqueeze(input,0)
     input = torch.unsqueeze(input,0)
     input = input.cuda().float()
     return input,im


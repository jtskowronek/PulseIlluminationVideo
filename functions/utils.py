import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import scipy.io as scio
import matplotlib.pyplot as plt


def meas2tensor(params):
     fullpath = params.input+params.name
     im = Image.open(fullpath)
     im = im.resize(size=(params.size[1],params.size[0]))
     im = np.array(im)
     inputT = torch.from_numpy(im)/255
     inputT = torch.unsqueeze(inputT,0)
     inputT = torch.unsqueeze(inputT,0)
     inputT = inputT.cuda().float()
     return inputT,im

def code2tensor(params):

    code = np.array(params.code)
    code = torch.from_numpy(code)

    mask = torch.ones(size=(params.size[0],params.size[1],params.frames))
    mask = mask*code
    mask = torch.permute(mask,(2,0,1))
    mask = torch.unsqueeze(mask,0).cuda().float()
    return(mask)



def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]  

    

def inputTensor(params):
    
    if params.inputType == 'noise':
        
        out = torch.rand(size=(params.frames,params.size[0],params.size[1]))
        out = torch.unsqueeze(out,0)
        out = out.cuda().float()
        
    if params.inputType == 'meas':    
        
        out,img =  meas2tensor(params)
        
    if params.inputType == 'hybrid':   
        
        out1 = torch.rand(size=(params.frames,params.size[0],params.size[1]))
        out1 = torch.unsqueeze(out1,0)
        out1 = torch.unsqueeze(out1,0)
        out1 = out1.cuda().float()
        
        out2,img =  meas2tensor(params)
        
        out = out1*0.5+out2*0.5
        
        
    return out[0,:]
        
        
        
def save2Mat(datacube,meas,gt,params):        
        
    vid = torch.permute(datacube,(2,3,1,0))     
    vid = torch.squeeze(vid)
    vid = vid.detach().cpu().numpy()
    
    gtm = torch.squeeze(gt)
    gtm = gtm.detach().cpu().numpy()
    
    measo = torch.squeeze(meas)
    measo = measo.detach().cpu().numpy()
    
    scio.savemat(params.output + "output.mat", {'vid': vid,'gt':gtm,'meas':measo})   



def plotTensor(t):
    t = torch.squeeze(t)
    t = t.detach().cpu().numpy()
    plt.imshow(t)
    plt.show()    
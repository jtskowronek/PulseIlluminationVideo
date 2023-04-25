import torch
import torch.nn as nn
from models.skip import skip

class FrameNet(nn.Module):

    def __init__(self, num_frames=16):
        super(FrameNet, self).__init__()

        self.FirstStep = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.StepDown = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.StepUp = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        

        self.LastStep = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())
        

        self.Encoder = []
        self.Encoder.append(self.FirstStep)
        for i in range(4):
            self.Encoder.append(self.StepDown)
        self.Encoder = nn.Sequential(*self.Encoder)    

        self.Decoder = []
        for i in range(4):
            self.Decoder.append(self.StepUp) 
        self.Decoder.append(self.LastStep) 
        self.Decoder = nn.Sequential(*self.Decoder)  


        self.UnetModule = nn.Sequential( 
            self.Encoder, 
            self.Decoder) 
        
        
        net = skip(1, 1, 
               num_channels_down = [64] * 1,
               num_channels_up =   [64] * 1,
               num_channels_skip =    [64] * 1,  
               filter_size_up = 3, filter_size_down = 3, 
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, act_fun='LeakyReLU')
        
        self.layers = nn.ModuleList()
        for i in range(num_frames):
            self.layers.append(net)
            
        self.frameBuilds = nn.ModuleList()
        for i in range(num_frames):
            self.frameBuilds.append(net)    
        
        

    def forward(self, y):
        inT = torch.unsqueeze(y[:,0,:,:],1)
        out = torch.zeros(size=y.shape,device=y.device)
        for i in range(len(self.layers)):
            inT = self.layers[i](inT+torch.unsqueeze(y[:,i,:,:],1))
            f = self.frameBuilds[i](inT)
            out[:,i,:,:] = f 

        return out



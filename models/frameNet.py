import torch
import torch.nn as nn


class FrameNet(nn.Module):

    def __init__(self, num_block=16):
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
            nn.MaxPool2d(stride=2))
        
        self.StepDown = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLUReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLUReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(stride=2))
        
        self.StepUp = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLUReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLUReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        

        self.LastStep = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.nn.Sigmoid())
        

        self.Encoder = nn.ModuleList()
        self.Encoder.append(self.FirstStep)
        for i in range(3):
            self.Encoder.append(self.StepDown)

        self.Decoder = nn.ModuleList()
        for i in range(3):
            self.Decoder.append(self.StepUp) 
        self.Decoder.append(self.LastStep) 

        self.UnetModule = nn.ModuleList()  
        self.UnetModule.append(self.Encoder) 
        self.UnetModule.append(self.Decoder) 


    def forward(self, y):
        out = self.UnetModule(y)

        return out



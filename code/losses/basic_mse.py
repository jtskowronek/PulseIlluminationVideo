import torch
import torch.nn as nn

class Loss(nn.Module):
    
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, inputs, targets):        
        tmp = (inputs-targets)**2
        loss =  torch.sqrt(torch.mean(tmp) )       
        return loss
    
    
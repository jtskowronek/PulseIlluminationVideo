import matplotlib.pyplot as plt

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *
from functions import *

import torch
import torch.optim

from skimage.measure import compare_psnr

from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor



parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--iter', default=2000, type=int, help='max epoch')
parser.add_argument('--LR', default=0.01, type=float)
parser.add_argument('--Fr', default=16, type=int, help='compressive rate')
parser.add_argument('--size', default=[256, 340], type=int, help='input image resolution')
parser.add_argument('--Fr', default=16, type=int, help='compressive rate')
parser.add_argument('--code', default=[1,0,1,1,1,0,0,0,1,0,1,1,0,1,1,1], type=int, help='Code')


args = parser.parse_args()


loss = nn.MSELoss()
loss.cuda()



import matplotlib.pyplot as plt

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from models.downsampler import Downsampler

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


imsize = -1 
factor = 4 # 8
PLOT = True

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8 
path_to_image = 'data/sr/zebra_GT.png'
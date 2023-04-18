from models.unet import UNet
from models.skip import skip
from models.revsci import re_3dcnn
from models.s2tnet import s2tnet



def getnetwork(params):

    if params.network == '3DUNet':

        net = re_3dcnn(num_block=8,num_group=2)

    if params.network == 'skip':

        net = skip(params.frames, params.frames, 
               num_channels_down = [128] * 5,
               num_channels_up =   [128] * 5,
               num_channels_skip =    [128] * 5,  
               filter_size_up = 3, filter_size_down = 3, 
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, act_fun='LeakyReLU')
   

    if params.network == 's2tnet':
       net = s2tnet(in_chans=params.frames,output_chans=params.frames)

    return net

from models.unet import UNet
from models.skip import skip
from models.revsci import re_3dcnn
from models.unet3d import UNet3D, ResidualUNet3D, ResidualUNetSE3D



def getnetwork(params):

    if params.network == 'RevSCI':

        net = re_3dcnn(num_block=8,num_group=2)

    if params.network == 'skip':

        net = skip(params.frames, params.frames, 
               num_channels_down = [128] * 5,
               num_channels_up =   [128] * 5,
               num_channels_skip =    [128] * 5,  
               filter_size_up = 3, filter_size_down = 3, 
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, act_fun='LeakyReLU')
   

    if params.network == '3DUNet':
       net = UNet3D(in_channels=1,
                      out_channels=1,
                      num_groups = 4,
                      final_sigmoid = True,
                      is_segmentation=False)

    return net


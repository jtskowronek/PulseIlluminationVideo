from models.skip import skip
from models.revsci import re_3dcnn
from models.frameNet import FrameNet
from models.HQS_temporal_3DConv_RFMb_HSA import HQSNet


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
   

    if params.network == 'SCI3D': 
        cnnpar = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')
        cnnpar.add_argument('--layer_num', type=int, default=10, help='phase number of HQS-Net')
        cnnpar.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
        cnnpar.add_argument('--temporal_patch', type=int, default=3,
                    help='the number of frames where temporal step takes as input')
        model = HQSNet(cnnpar)

    if params.network == 'FrameNet':
       net = FrameNet()   

    return net


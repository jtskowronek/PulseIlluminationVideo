import os.path as osp
from config.config_file import Config
import os
import torch


def update_cfg(args):
    cfg = Config.fromfile(args.config)
    cfg.resize_h,cfg.resize_w = args.resolution
    cfg.crop_h,cfg.crop_w = args.dataset_crop
    
    cfg.train_pipeline[4]['resize_h'],cfg.train_pipeline[4]['resize_w'] = args.resolution
    cfg.train_pipeline[1]['crop_h'],cfg.train_pipeline[1]['crop_w'] = args.dataset_crop
    cfg.train_data['data_root'] = args.train_dataset_path
    cfg.test_data['data_root'] = args.test_dataset_path
    cfg.train_data.mask_path = args.mask_path
    cfg.train_data.mask_shape = (args.resolution[0],args.resolution[1],args.frames)
    
    cfg.save_image_config['interval'] = args.saveImageEach
    cfg.runner['max_epoch'] = args.Epochs
    
    cfg.checkpoints = args.checkpoints
    cfg.checkpoint_config['interval'] = args.saveModelEach
    cfg.optimizer['lr'] = args.learning_rate
    cfg.data['samples_per_gpu'] = args.batch_size
    
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(args.config))[0])

    if args.resume is not None:
        cfg.resume = args.resume



    cfg.log_dir = osp.join("./train_results/"+args.work_dir,"log")
    cfg.show_dir = osp.join("./train_results/"+args.work_dir,"show")
    cfg.train_image_save_dir = osp.join("./train_results/"+args.work_dir,"train_images")
    cfg.checkpoints_dir = osp.join("./train_results/"+args.work_dir,"checkpoints")

    if not osp.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not osp.exists(cfg.show_dir):
        os.makedirs(cfg.show_dir)
    if not osp.exists(cfg.train_image_save_dir):
        os.makedirs(cfg.train_image_save_dir)
    if not osp.exists(cfg.checkpoints_dir):
        os.makedirs(cfg.checkpoints_dir)    
    return cfg


def subsample_tensor(tensor, block_size):
    # tensor: Input tensor (shape: [batch_size, channels, height, width])
    # block_size: Integer (either 2 or 4), specifies the subsample size.

    if block_size not in [2, 4, 8]:
        raise ValueError("block_size should be either 2 or 4 or 8.")

    # Get the dimensions of the tensor
    batch_size, channels, height, width = tensor.size()

    # Calculate the new dimensions after subsampling
    new_height = height // block_size
    new_width = width // block_size

    # Reshape the tensor into blocks for subsampling
    subsampled_tensor = tensor.view(batch_size, channels, new_height, block_size, new_width, block_size)

    # Reshape and transpose to create patches
    subsampled_tensor = subsampled_tensor.permute(0, 1, 3, 5, 2, 4).contiguous()
    subsampled_tensor = subsampled_tensor.view(batch_size * (block_size ** 2), channels, new_height, new_width)

    return subsampled_tensor



def reconstruct_tensor(subsampled_tensor, original_shape, block_size):
    # subsampled_tensor: Input tensor after subsampling
    # original_shape: Tuple (batch_size, channels, original_height, original_width)
    # block_size: Integer (either 2 or 4), specifies the subsample size.

    if block_size not in [2, 4, 8]:
        raise ValueError("block_size should be either 2 or 4 or 8.")

    # Get the original dimensions
    batch_size, channels, original_height, original_width = original_shape

    # Calculate the new dimensions after subsampling
    new_height = original_height // block_size
    new_width = original_width // block_size

    # Reshape the tensor into blocks for reconstruction
    reconstructed_tensor = subsampled_tensor.view(batch_size, block_size, block_size, channels, new_height, new_width)

    # Reshape and transpose to create patches
    reconstructed_tensor = reconstructed_tensor.permute(0, 3, 4, 1, 5, 2).contiguous()
    reconstructed_tensor = reconstructed_tensor.view(batch_size, channels, original_height, original_width)

    return reconstructed_tensor
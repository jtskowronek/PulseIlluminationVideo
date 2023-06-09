
from config.config_file import Config



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
    return cfg
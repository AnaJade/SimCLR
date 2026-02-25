import argparse
import pathlib
import pynvml
import random
import socket
import sys
from sys import platform
import torch
import torch.backends.cudnn as cudnn
from addict import Dict
from torchvision import models
import torchvision.transforms as transforms

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import FeatureModelSimCLR
from simclr import SimCLR

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import OCTDataset, build_image_root

# Img size and moco_dim (nb of classes) values based on the dataset
img_size_dict = {'stl10': 96,
                 'cifar10': 32,
                 'cifar100': 32}
num_cluster_dict = {'stl10': 10,
                    'cifar10': 10,
                    'cifar100': 100}

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [0.485, 0.456, 0.406]
mean['npy'] = [0.485, 0.456, 0.406]
mean['npy224'] = [0.485, 0.456, 0.406]
# mean['oct'] = [149.888, 149.888, 149.888]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
std['npy'] = [0.229, 0.224, 0.225]
std['npy224'] = [0.229, 0.224, 0.225]
# std['oct'] = [11.766, 11.766, 11.766]

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    help='Path to the config file',
                    type=str)

def main():
    args = parser.parse_args()
    if args.config_path is None:
        args.config_path = pathlib.Path('../config.yaml')
    config_file = pathlib.Path(args.config_path)

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    configs = utils.load_configs(config_file)
    if platform == "linux" or platform == "linux2":
        if 'hpc' in socket.gethostname() or 'u00' in socket.gethostname():
            dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_hpc'])
        else:
            dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_linux'])
    elif platform == "win32":
        dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_windows'])
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    pre_processing = Dict(configs['data']['pre_processing'])
    use_mini_dataset = configs['data']['use_mini_dataset']
    mean['oct'] = 3 * [configs['data']['img_mean'] / 255]
    std['oct'] = 3 * [configs['data']['img_std'] / 255]
    img_size_dict['oct'] = (512, ascan_per_group)
    num_cluster_dict['oct'] = len(labels)

    ### Convert config file values to the args variable equivalent (match the format of the existing code)
    print("Assigning config values to corresponding args variables...")
    # Dataset
    args.dataset_name = configs['SimCLR']['dataset_name']
    args.scan_no_noise = configs['data']['pre_processing']['no_noise']  # Add to args for logging
    args.scan_use_movmean = configs['data']['pre_processing']['use_movmean']  # Add to args for logging
    args.scan_use_speckle = configs['data']['pre_processing']['use_speckle']  # Add to args for logging
    args.scan_sampling = configs['data']['pre_processing']['ascan_sampling']  # Add to args for logging
    dataset_root = pathlib.Path(dataset_path).joinpath(
        'OCT_lab_data' if args.dataset_name == 'oct' else args.dataset_name)
    image_root = build_image_root(ascan_per_group, pre_processing)
    print(f"dataset image root: {dataset_root.joinpath(image_root)}")
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    args.map_df_paths = {
        split: dataset_root.joinpath(image_root).joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}
    args.img_channel = configs['SimCLR']['img_channel']
    if args.dataset_name != 'oct':
        args.img_channel = 3
    args.sample_within_image = configs['SimCLR']['sample_within_image']
    args.img_reshape = configs['SimCLR']['img_reshape']
    if args.img_reshape is not None:
        args.img_size = args.img_reshape
    else:
        args.img_size = 512 # img_size_dict[args.dataset_name]
    args.use_iipp = configs['SimCLR']['use_iipp']
    args.num_same_area = configs['SimCLR']['num_same_area']
    args.use_simclr_augmentations = configs['SimCLR']['use_simclr_augmentations']
    args.ascan_per_group = ascan_per_group

    # Training params
    args.seed = configs['training']['random_seed']
    args.dataset_sample = configs['SimCLR']['dataset_sample']
    args.arch = configs['SimCLR']['arch']
    args.use_pretrained = configs['SimCLR']['use_pretrained']
    args.workers = configs['SimCLR']['num_workers']
    args.epochs = configs['SimCLR']['max_epochs']
    args.batch_size = configs['SimCLR']['batch_size']
    args.lr = configs['SimCLR']['lr']
    args.weight_decay = configs['SimCLR']['weight_decay']
    args.disable_cuda = configs['SimCLR']['disable_cuda']
    args.fp16_precision = configs['SimCLR']['fp16_precision']
    args.out_dim = num_cluster_dict[args.dataset_name]
    args.log_every_n_steps = configs['SimCLR']['log_every_n_steps']
    args.temperature = configs['SimCLR']['temperature']
    args.n_views = 2
    args.gpu_index = configs['SimCLR']['gpu_index']
    args.patience = configs['SimCLR']['patience']
    if (platform == "linux" or platform == "linux2") and ('hpc' in socket.gethostname() or 'u00' in socket.gethostname()):
        print(f"socket name: {socket.gethostname()}")
        args.save_folder = pathlib.Path(r'/fibus/fs0/14/cab8351/OCT_classification/SimCLR').joinpath(f'weights_{args.arch}')
    else:
        args.save_folder = pathlib.Path().resolve().joinpath(f'weights_{args.arch}')
    if not args.save_folder.is_dir():
        args.save_folder.mkdir(parents=True)

    args.wandb = Dict()
    args.wandb.wandb_log = configs['wandb']['wandb_log']
    args.wandb.project_name = configs['wandb']['project_name']
    if args.wandb.project_name != 'Test-project':
        args.wandb.project_name = 'OCT_SimCLR'

    ###############################################
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # Set all random seeds
    print("Setting random seed...")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        # print('__CUDNN VERSION:', torch.backends.cudnn.version())
        # print('__Number CUDA Devices:', torch.cuda.device_count())
        args.device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.deterministic = True
        cudnn.benchmark = True
        print('Selected GPU index:', args.gpu_index)
        print('__CUDA Device Name:', torch.cuda.get_device_name(args.gpu_index))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(args.gpu_index).total_memory / 1e9)
        print('Clearing cache...')
        torch.cuda.empty_cache()
        print('__CUDA Device Reserved Memory [GB]:', torch.cuda.memory_reserved(args.gpu_index) / 1e9)
        print('__CUDA Device Allocated Memory [GB]:', torch.cuda.memory_allocated(args.gpu_index) / 1e9)
        print('Stats with pynvml:')
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        print(f'total    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')
        pynvml.nvmlShutdown()
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.dataset_name == 'oct':
        img_transforms = [transforms.ToTensor(),  # scales pixel values to [0, 1]
                          transforms.Resize((args.img_reshape, args.img_reshape)),
                          transforms.Normalize(mean=mean[args.dataset_name],
                                               std=std[args.dataset_name])]
        if (args.sample_within_image <= 0) and (args.img_reshape <= 480):
            img_transforms.insert(1, transforms.CenterCrop(480))
        if args.img_channel == 1:
            img_transforms.append(transforms.Grayscale())
        if not args.use_simclr_augmentations:
            aug = [transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.3),  # Used to counter flipped scans
                   transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
                   transforms.RandomApply([transforms.RandomRotation(degrees=8),
                                           # transforms.CenterCrop(size=(188, 236)), # Used in the paper, but not really applicable here
                                           transforms.RandomHorizontalFlip()], p=0.5),
                   ]
            img_transforms = img_transforms + aug
        oct_args = {'map_df_paths': args.map_df_paths,
                    'labels_dict': args.labels_dict,
                    'img_size': args.img_size,
                    'img_channel': args.img_channel,
                    'sample_within_image': args.sample_within_image,
                    'use_iipp': args.use_iipp,
                    'num_same_area': args.num_same_area,
                    'use_simclr_augmentations': args.use_simclr_augmentations,
                    'transforms_list': img_transforms,
                    'dataset_sample': args.dataset_sample}
        dataset = ContrastiveLearningDataset(dataset_root)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, oct_args)
    else:
        dataset = ContrastiveLearningDataset(dataset_root)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    if args.use_iipp:
        batch_size = int(round(args.batch_size / args.num_same_area))
    else:
        batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    feature_model = FeatureModelSimCLR(arch=args.arch,
                                       out_dim=args.out_dim,
                                       pretrained=args.use_pretrained,
                                       img_channel=args.img_channel)

    optimizer = torch.optim.Adam(feature_model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=feature_model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()

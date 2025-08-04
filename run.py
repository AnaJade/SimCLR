import argparse
import pathlib
import sys
from sys import platform
import torch
import torch.backends.cudnn as cudnn
from torchvision import models

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import OCTDataset

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
mean['oct'] = [149.888, 149.888, 149.888]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
std['npy'] = [0.229, 0.224, 0.225]
std['npy224'] = [0.229, 0.224, 0.225]
std['oct'] = [11.766, 11.766, 11.766]

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

"""
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
"""

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
        dataset_root = pathlib.Path(configs['data']['dataset_root_linux'])
        dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_linux'])
    elif platform == "win32":
        dataset_root = pathlib.Path(configs['data']['dataset_root_windows'])
        dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_windows'])
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']
    # oct_img_root = pathlib.Path(f'OCT_lab_data/{ascan_per_group}mscans')
    img_size_dict['oct'] = (512, ascan_per_group)
    num_cluster_dict['oct'] = len(labels)

    ### Convert config file values to the args variable equivalent (match the format of the existing code)
    print("Assigning config values to corresponding args variables...")
    # Dataset
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    args.map_df_paths = {
        split: dataset_root.joinpath(f"{ascan_per_group}mscans").joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}

    args.dataset_name = configs['SimCLR']['dataset_name']
    args.data = pathlib.Path(dataset_path).joinpath('OCT_lab_data' if args.dataset_name == 'oct' else args.dataset_name)
    print(f"args.data: {args.data}")

    # args.all = configs['SPICE']['MoCo']['use_all']
    args.img_size = img_size_dict[args.dataset_name]

    # Training params
    args.seed = configs['training']['random_seed']
    # args.save_folder = pathlib.Path(configs['SPICE']['MoCo']['save_folder']).joinpath(args.dataset_name).joinpath('moco')
    # args.save_freq = configs['SPICE']['MoCo']['save_freq']
    args.arch = configs['SimCLR']['arch']
    args.workers = configs['SimCLR']['num_workers']
    args.epochs = configs['SimCLR']['max_epochs']
    # args.start_epoch = configs['SPICE']['MoCo']['start_epoch']
    args.batch_size = configs['SimCLR']['batch_size']
    args.lr = configs['SimCLR']['lr']
    # args.schedule = configs['SPICE']['MoCo']['lr_schedule']
    # args.momentum = configs['SPICE']['MoCo']['momentum']
    args.weight_decay = configs['SimCLR']['weight_decay']
    # args.print_freq = configs['SPICE']['MoCo']['print_freq']
    # args.resume = pathlib.Path(args.save_folder).joinpath('checkpoint_last.pth.tar') if configs['SPICE']['MoCo'][
    #     'resume'] else False
    args.disable_cuda = configs['SimCLR']['disable_cuda']
    args.fp16_precision = configs['SimCLR']['fp16_precision']
    args.out_dim = num_cluster_dict[args.dataset_name]
    args.log_every_n_steps = configs['SimCLR']['log_every_n_steps']
    args.temperature = configs['SimCLR']['temperature']
    args.n_views = 2
    args.gpu_index = configs['SimCLR']['gpu_index']

    ###############################################
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.dataset_name == 'oct':
        oct_args = {'map_df': args.map_df_paths,
                    'labels_dict': args.labels_dict,
                    'size': args.img_size}
        dataset = ContrastiveLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, oct_args)
    else:
        dataset = ContrastiveLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()

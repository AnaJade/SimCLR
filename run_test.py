import argparse
import pathlib
import sys
from sys import platform
import torch
import torch.backends.cudnn as cudnn
from addict import Dict
from torchvision import models

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import get_supervised_oct_data_loaders, get_stl10_data_loaders, build_image_root
from ssl_test import FeatureExtractor, LogisticRegressionEvaluator

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
mean['oct'] = [x /255 for x in [42.573, 42.573, 42.573]]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
std['npy'] = [0.229, 0.224, 0.225]
std['npy224'] = [0.229, 0.224, 0.225]
std['oct'] = [x /255 for x in [26.688, 26.688, 26.688]]

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
        dataset_root = pathlib.Path(configs['data']['dataset_root_linux'])
        dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_linux'])
    elif platform == "win32":
        dataset_root = pathlib.Path(configs['data']['dataset_root_windows'])
        dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_windows'])
    # chkpt_file = pathlib.Path('runs/Sep18_19-15-48_ilmare/checkpoint_0200.pth.tar')
    # chkpt_file = pathlib.Path('runs/Sep20_09-16-58_ilmare/checkpoint_best_top1.pth')
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
    args.data = pathlib.Path(dataset_path).joinpath(
        'OCT_lab_data' if args.dataset_name == 'oct' else args.dataset_name)
    image_root = build_image_root(ascan_per_group, pre_processing)
    print(f"dataset image root: {args.data.joinpath(image_root)}")
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    args.map_df_paths = {
        split: args.data.joinpath(image_root).joinpath(
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
        args.img_size = 512  # BYOL requires square images, so all images will be reshaped to 512x512
    args.use_iipp = configs['SimCLR']['use_iipp']
    args.num_same_area = configs['SimCLR']['num_same_area']
    args.use_simclr_augmentations = configs['SimCLR']['use_simclr_augmentations']
    args.ascan_per_group = ascan_per_group

    # Training params
    args.seed = configs['training']['random_seed']
    args.dataset_sample = configs['SimCLR']['dataset_sample']
    args.arch = configs['SimCLR']['arch']
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
    args.save_folder = pathlib.Path().resolve().joinpath(f'weights_{args.arch}')
    chkpt_file = list(args.save_folder.rglob('checkpoint_best*.pt'))[0]

    args.wandb = Dict()
    args.wandb.wandb_log = configs['wandb']['wandb_log']
    args.wandb.project_name = configs['wandb']['project_name']

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

    # Create train and test sets
    if args.dataset_name == 'oct':
        train_loader, valid_loader, test_loader = get_supervised_oct_data_loaders(args.data, args, args.batch_size,
                                                                                  mean=mean[args.dataset_name],
                                                                                  std=std[args.dataset_name],
                                                                                  shuffle=False)

    else:
        train_loader, test_loader = get_stl10_data_loaders(args.data, args.batch_size, shuffle=False, download=False)

    # Extract features
    print(f"Extracting features on the train and test sets...")
    feature_extractor = FeatureExtractor(args, 'simclr', chkpt_file)
    X_train_feature, y_train, X_test_feature, y_test = feature_extractor.get_features(train_loader, test_loader)

    # Train logistic regression
    print(f"Training the regression model...")
    log_regressor_evaluator = LogisticRegressionEvaluator(n_features=X_train_feature.shape[1], n_classes=args.out_dim, args=args)
    log_regressor_evaluator.train(X_train_feature, y_train, X_test_feature, y_test)


if __name__ == "__main__":
    main()

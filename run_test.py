import argparse
import pathlib
import sys
from sys import platform
import torch
import torch.backends.cudnn as cudnn
from addict import Dict
from torchvision import models

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.neighbors import KNeighborsClassifier
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import importlib.util
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, adjusted_rand_score, normalized_mutual_info_score

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


class FeatureExtractor(object):
    def __init__(self, args, ckp_file):
        self.ckp_file = ckp_file
        self.args = args
        self.model = FeatureModelSimCLR(arch=args.arch, out_dim=args.out_dim, pretrained=False, img_channel=args.img_channel)
        # FeatureModelSimCLR(base_model=args.arch, out_dim=args.out_dim)

        # Load weights
        state_dict = torch.load(self.ckp_file, map_location=self.args.device)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(args.device)

    def _inference(self, loader):
        feature_vector = []
        labels_vector = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.args.device)
            labels_vector.extend(batch_y)

            features = self.model(batch_x)
            feature_vector.extend(features.cpu().detach().numpy())

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)

        print("Features shape {}".format(feature_vector.shape))
        return feature_vector, labels_vector

    def get_resnet_features(self, train_loader, test_loader):
        X_train_feature, y_train = self._inference(train_loader)
        X_test_feature, y_test = self._inference(test_loader)

        return X_train_feature, y_train, X_test_feature, y_test


class LogisticRegression(nn.Module):

    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(n_features, n_classes)
        # self.model = nn.Sequential(nn.Linear(n_features, n_classes),
        #                            nn.Linear(n_classes, n_classes))

    def forward(self, x):
        return self.model(x)

class LogiticRegressionEvaluator(object):
    def __init__(self, n_features, n_classes, args):
        self.args = args
        self.log_regression = LogisticRegression(n_features, n_classes).to(self.args.device)
        self.scaler = preprocessing.StandardScaler()

    def _normalize_dataset(self, X_train, X_test):
        print("Standard Scaling Normalizer")
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def eval(self, test_loader):
        correct = 0
        total = 0

        logits_epoch = []
        y_true_epoch = []
        with torch.no_grad():
            self.log_regression.eval()
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                logits = self.log_regression(batch_x)

                predicted = torch.argmax(logits, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                # Save valuesbit
                logits_epoch.append(logits)
                y_true_epoch.append(batch_y)

            final_acc = 100 * correct / total
            logits_epoch = torch.concat(logits_epoch)
            y_true_epoch = torch.concat(y_true_epoch)
            self.log_regression.train()
            return final_acc, logits_epoch, y_true_epoch

    def create_data_loaders_from_arrays(self, X_train, y_train, X_test, y_test):
        X_train, X_test = self._normalize_dataset(X_train, X_test)

        train = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                               torch.from_numpy(y_train).type(torch.long))
        train_loader = torch.utils.data.DataLoader(train, batch_size=396, shuffle=False)

        test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).type(torch.long))
        test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
        return train_loader, test_loader

    def train(self, X_train, y_train, X_test, y_test):

        train_loader, test_loader = self.create_data_loaders_from_arrays(X_train, y_train, X_test, y_test)

        weight_decay = self._sample_weight_decay()

        optimizer = torch.optim.Adam(self.log_regression.parameters(), 3e-4, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        best_nmi = 0
        best_epoch_acc = 0
        best_epoch_precision = 0
        best_epoch_recall = 0
        best_epoch_ari = 0
        best_epoch = 0
        print("Training regression model...")
        for e in tqdm(range(200)):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                optimizer.zero_grad()
                logits = self.log_regression(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            acc, logits_epoch, y_true_epoch = self.eval(test_loader)

            # Get other metrics
            preds_epoch = logits_epoch.argmax(1)
            eval_df_epoch = pd.DataFrame(torch.vstack((y_true_epoch.cpu(), preds_epoch.cpu())).T, columns=['label', 'pred'])
            # Calculate other metrics
            # Precision and recall
            # https://medium.com/data-science-in-your-pocket/calculating-precision-recall-for-multi-class-classification-9055931ee229
            # Precision (tp/(tp+fp)) and recall (tp/(tp+fn)), macro gives better results as micro
            precision = precision_score(eval_df_epoch['label'], eval_df_epoch['pred'], average='macro', zero_division=0)
            recall = recall_score(eval_df_epoch['label'], eval_df_epoch['pred'], average='macro')
            # Adjusted rand index (ARI)
            ari = adjusted_rand_score(eval_df_epoch['label'], eval_df_epoch['pred'])
            # Normalized mutual information (NMI)
            nmi = normalized_mutual_info_score(eval_df_epoch['label'], eval_df_epoch['pred'])
            if nmi > best_nmi:
                # print("Saving new model with accuracy {}".format(epoch_acc))
                best_nmi = nmi
                best_epoch = e
                best_epoch_acc = acc
                best_epoch_precision = precision
                best_epoch_recall = recall
                best_epoch_ari = ari
                torch.save(self.log_regression.state_dict(), 'log_regression.pth')

        print("--------------")
        print("Done training")
        print(f"Best nmi @ epoch {best_epoch}: {best_nmi}")
        print(f"Accuracy @ epoch {best_epoch}: {best_epoch_acc}")
        print(f"Precision @ epoch {best_epoch}: {best_epoch_precision}")
        print(f"Recall @ epoch {best_epoch}: {best_epoch_recall}")
        print(f"ARI @ epoch {best_epoch}: {best_epoch_ari}")

def get_stl10_data_loaders(root_path, batch_size=128, shuffle=False, download=False):
    train_dataset = datasets.STL10(root_path, split='train', download=download,
                                   transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.STL10(root_path, split='test', download=download,
                                  transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def get_oct_data_loaders(root_path:pathlib.Path, args: argparse.Namespace, batch_size:int, shuffle=False):
    img_transforms = [transforms.ToTensor(),
                      transforms.Resize((args.img_reshape, args.img_reshape)),
                      transforms.Normalize(mean=mean[args.dataset_name],
                                           std=std[args.dataset_name])]
    if args.img_channel == 1:
        img_transforms.append(transforms.Grayscale())
    img_transforms = transforms.Compose(img_transforms)
    train_dataset = OCTDataset(root_path, 'train',
                               args.map_df_paths, args.labels_dict,
                               ch_in=args.img_channel,
                               sample_within_image=args.sample_within_image,
                               use_iipp=False, # args.use_iipp,
                               num_same_area=-1,
                               transforms=img_transforms,
                               pre_sample=args.dataset_sample)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = OCTDataset(root_path, 'test',
                              args.map_df_paths, args.labels_dict,
                              ch_in=args.img_channel,
                              sample_within_image=args.sample_within_image,
                              use_iipp=False,
                              num_same_area=-1,
                              transforms=img_transforms,
                              pre_sample=args.dataset_sample)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


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
        train_loader, test_loader = get_oct_data_loaders(args.data, args, args.batch_size, shuffle=False)

    else:
        train_loader, test_loader = get_stl10_data_loaders(args.data, args.batch_size, shuffle=False, download=False)

    # Extract features
    print(f"Extracting features on the train and test sets...")
    feature_extractor = FeatureExtractor(args, chkpt_file)
    X_train_feature, y_train, X_test_feature, y_test = feature_extractor.get_resnet_features(train_loader, test_loader)

    # Train logistic regression
    print(f"Training the regression model...")
    log_regressor_evaluator = LogiticRegressionEvaluator(n_features=X_train_feature.shape[1], n_classes=args.out_dim, args=args)
    log_regressor_evaluator.train(X_train_feature, y_train, X_test_feature, y_test)

if __name__ == "__main__":
    main()

import pathlib
import sys
import timm
import torch
import torch.nn as nn
import torchvision.models as models

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils


class FeatureModelSimCLR(nn.Module):
    def __init__(self, arch, out_dim, pretrained, img_channel):
        super().__init__()
        self.pretrained = pretrained
        self.backbone = self._get_basemodel(arch)

        # Adjust first layer if grayscale is used
        if img_channel == 1:
            self.backbone = utils.update_backbone_channel(self.backbone, img_channel)
            """
            if 'resnet' in arch:
                self.backbone.conv1.in_channels = 1
                self.backbone.conv1.weight = nn.Parameter(
                    torch.mean(self.backbone.conv1.weight, dim=1, keepdim=True))
            if 'vit' in arch:
                self.backbone.conv_proj.in_channels = 1
                self.backbone.conv_proj.weight = nn.Parameter(
                    torch.mean(self.backbone.conv_proj.weight, dim=1, keepdim=True))
            if 'efficientnet' in arch:
                pass
            if 'swin' in arch:
                pass
            if 'convnext' in arch:
                pass
            """

        # add mlp projection head
        if 'resnet' in arch:
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                             nn.ReLU(),
                                             nn.Linear(dim_mlp, out_dim))
        if 'vit' in arch:
            dim_mlp = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                     nn.ReLU(),
                                                     nn.Linear(dim_mlp, out_dim))
        if 'efficientnet' in arch:
            dim_mlp = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                list(self.backbone.classifier)[0],
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, out_dim))
        if 'swin' in arch:
            dim_mlp = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                               nn.ReLU(),
                                               nn.Linear(dim_mlp, out_dim))
        if 'convnext' in arch:
            dim_mlp = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Sequential(
                *list(self.backbone.classifier)[:2],
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, out_dim))
        if 'pvt' in arch:
            dim_mlp = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                               nn.ReLU(),
                                               nn.Linear(dim_mlp, out_dim))

    def _get_basemodel(self, model_name):
        weights = 'DEFAULT' if self.pretrained else None
        if model_name == 'resnet18':
            feature_model = models.resnet18(weights=weights)
        elif model_name == 'resnet50':
            feature_model = models.resnet50(weights=weights)
        elif model_name == 'vitb16':
            feature_model = models.vit_b_16(weights=weights)
        elif model_name == 'efficientnetV2s':
            feature_model = models.efficientnet_v2_s(weights=weights)
        elif model_name == 'swinv2t':
            feature_model = models.swin_v2_t(weights=weights)
        elif model_name == 'convnextt':
            feature_model = models.convnext_tiny(weights=weights)
        elif model_name == 'pvtv2b0':
            feature_model = timm.create_model("pvt_v2_b0", pretrained=self.pretrained)
        else:
            feature_model = None
        return feature_model

    def forward(self, x):
        return self.backbone(x)
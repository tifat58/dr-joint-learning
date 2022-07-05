import torch
import torch.nn as nn
from torchvision import models
from self_attention_cv import ResNet50ViT, ViT
from modules.GAIN import GAIN

from utils.func import print_msg, select_out_features


def generate_gain_model(cfg):
    out_features = select_out_features(
        cfg.data.num_classes,
        cfg.train.criterion
    )

    model = build_model(
        cfg.train.network,
        out_features,
        cfg.train.pretrained
    )

    # if cfg.train.checkpoint:
    #     weights = torch.load(cfg.train.checkpoint)
    #     model.load_state_dict(weights, strict=True)
    #     print_msg('Load weights form {}'.format(cfg.train.checkpoint))

    if cfg.base.device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(cfg.base.device)
    return model


# def build_gain_model(network, num_classes, pretrained=False):
#
#     if 'resnet50_vit' in network:
#         model = ResNet50ViT(img_dim=256, pretrained_resnet=True,
#                             blocks=6, num_classes=num_classes,
#                             dim_linear_block=64, dim=64)
#         print("vit resnet50 loaded")
#         return model
#
#     model = BUILDER[network](pretrained=pretrained)
#     if 'resnet' in network or 'resnext' in network or 'shufflenet' in network:
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#
#     elif 'densenet' in network:
#         model.classifier = nn.Linear(model.classifier.in_features, num_classes)
#     elif 'vgg' in network:
#         model.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )
#     elif 'mobilenet' in network:
#         model.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(model.last_channel, num_classes),
#         )
#     elif 'squeezenet' in network:
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Conv2d(512, num_classes, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#     else:
#         raise NotImplementedError('Not implemented network.')
#
#     return model


def generate_model(cfg):
    out_features = select_out_features(
        cfg.data.num_classes,
        cfg.train.criterion
    )

    model = build_model(
        cfg.train.network,
        out_features,
        cfg.train.pretrained
    )

#     model.load_state_dict(torch.load('/home/hnguyen/DATA/SSL-Medical/KOTORI/Eye_Dataset/dr-joint-learning/classify_no_train/final_weights.pt'), strict=True) # specify weight path in the config file for models
    if cfg.train.checkpoint:
        try:
            weights = torch.load(cfg.train.checkpoint)
            model.load_state_dict(weights, strict=False)
        except:
            weights = torch.load(cfg.train.checkpoint)
            trunk_weights = weights["classy_state_dict"]["base_model"]["model"]["trunk"]
            prefix = "_feature_blocks."
            trunk_weights = {k[len(prefix):] : w for k, w in trunk_weights.items()}
            model.load_state_dict(trunk_weights, strict=False)
        print_msg('Load weights form {}'.format(cfg.train.checkpoint))

    if cfg.base.device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(cfg.base.device)
    return model


def build_model(network, num_classes, pretrained=False):

    if 'gain' in network:
        model = GAIN(grad_layer='layer4', num_classes=num_classes)

        print('Resnet Attention model loaded!!!')
        return model

    if 'resnet50_vit' in network:
        model = ResNet50ViT(img_dim=256, pretrained_resnet=True,
                            blocks=6, num_classes=num_classes,
                            dim_linear_block=64, dim=64)
        print("vit resnet50 loaded")
        return model
    
    if 'vit' in network:
        model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=num_classes,dim=512)
        print("vit model loaded")
        return model

    model = BUILDER[network](pretrained=pretrained)
    if 'resnet' in network or 'resnext' in network or 'shufflenet' in network:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif 'densenet' in network:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'vgg' in network:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif 'mobilenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
    elif 'squeezenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    else:
        raise NotImplementedError('Not implemented network.')

    return model


BUILDER = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
}

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model
from torchvision.models import resnet50
from collections import OrderedDict


class GAIN(nn.Module):
    def __init__(self, grad_layer, num_classes):
        super(GAIN, self).__init__()
        # self.model = make_model(
        #     model_name='resnet50',
        #     pretrained=True,
        #     num_classes=num_classes
        # )
        self.res_model = resnet50(pretrained=True)
        self.res_model.fc = nn.Linear(self.res_model.fc.in_features, num_classes)
        # print(self.res_model)
        # weights = torch.load('/mnt/sda/haal02-data/results/classification_results/resnet50_full_fgadr_playground/saves/best_validation_weights.pt')
        #
        # weights2 = OrderedDict()
        # for key, value in weights.items():
        #     new_key = 'res_model.' + key
        #     weights2[new_key] =value


        # self.res_model.load_state_dict(weights2, strict=False)


        grad_layer = 'layer4'
        self.grad_layer = grad_layer

        self.num_classes = num_classes

        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None

        # Register hooks
        self._register_hooks(grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100
        self.thershold = nn.Threshold(0.4, 0, inplace=False)
        self.mask_relu = nn.ReLU()

    def _register_hooks(self, grad_layer):
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in self.res_model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_classes), requires_grad=True)
        for i, label in enumerate(labels):
            ohe[i, label] = 1

        ohe = torch.autograd.Variable(ohe)

        return ohe

    def forward(self, images, labels):

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to unstable validation score. It is better to visualize attention maps at the testset

        is_train = self.res_model.training

        with torch.enable_grad():
            # labels_ohe = self._to_ohe(labels).cuda()
            # labels_ohe.requires_grad = True

            _, _, img_h, img_w = images.size()

            self.res_model.train(True)
            logits = self.res_model(images)  # BS x num_classes
            self.res_model.zero_grad()

            if not is_train:
                pred = F.softmax(logits).argmax(dim=1)
                labels_ohe = self._to_ohe(pred).cuda()
            else:
                labels_ohe = self._to_ohe(labels).cuda()

            gradient = logits * labels_ohe
            # print(gradient.shape, labels_ohe.shape)
            grad_logits = (logits * labels_ohe).sum()  # BS x num_classes
            grad_logits.backward(retain_graph=True)
            self.res_model.zero_grad()

        if is_train:
            self.res_model.train(True)
        else:
            self.res_model.train(False)
            self.res_model.eval()
            logits = self.res_model(images)

        backward_features = self.backward_features  # BS x C x H x W
        # bs, c, h, w = backward_features.size()
        # wc = F.avg_pool2d(backward_features, (h, w), 1)  # BS x C x 1 x 1

        """
        The wc shows how important of the features map
        """

        # Eq 2
        fl = self.feed_forward_features  # BS x C x H x W
        # print(fl.shape)
        # bs, c, h, w = fl.size()
        # fl = fl.view(1, bs * c, h, w)

        """
        fl is the feature maps during feed-forward
        """

        """
        We do 2d convolution to find the Attention maps. We consider wc as a filter matrix.
        """

        # Ac = F.relu(F.conv2d(fl, wc, groups=bs))
        # # Resize to be as same as of image size
        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        # Ac = Ac.permute((1, 0, 2, 3))
        # heatmap = Ac

        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)

        max_val = torch.max(Ac)
        if max_val == 0.0:
            max_val = max_val + 0.00001  # adding smoth
        if torch.isnan(max_val):
            print('max val is nan')

        Ac = Ac / max_val
        if torch.isnan(Ac.sum()):
            print('There is a nan value in heatmap')

        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        Ac = F.upsample_bilinear(Ac, size=images.size()[2:])

        heatmap = Ac
        # heatmap = self.thershold(heatmap)

        """
        Generate the soft-mask
        """

        Ac_min = Ac.min()
        Ac_max = Ac.max()
        scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)
        mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma))
        masked_image = images - images * mask

        logits_am = self.res_model(masked_image)

        return logits, logits_am, heatmap
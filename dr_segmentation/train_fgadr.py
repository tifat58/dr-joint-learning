#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from torch.autograd import Variable
import os
import numpy as np
import random
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import config_fgadr as config
from unet import UNet
from utils import get_images, get_images_fgadr, get_images_fgadr_from_pd
from dataset import IDRIDDataset
from fgadr_dataset import FGADRDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import argparse
import segmentation_models_pytorch as smp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rotation_angle = config.ROTATION_ANGEL
image_size = config.IMAGE_SIZE
image_dir = config.IMAGE_DIR
batchsize = config.TRAIN_BATCH_SIZE
dataset_name = config.DATASET_NAME
tatl = config.TATL


def eval_model(model, eval_loader):
    model.eval()
    masks_soft = []
    masks_hard = []

    with torch.set_grad_enabled(False):
        for inputs, true_masks in eval_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)
            bs, _, h, w = inputs.shape
            # not ignore the last few patches
            h_size = (h - 1) // image_size + 1
            w_size = (w - 1) // image_size + 1
            masks_pred = torch.zeros(true_masks.shape).to(dtype=torch.float)

            for i in range(h_size):
                for j in range(w_size):
                    h_max = min(h, (i + 1) * image_size)
                    w_max = min(w, (j + 1) * image_size)
                    inputs_part = inputs[:, :, i * image_size:h_max, j * image_size:w_max]
                    masks_pred_single = model(inputs_part)
                    masks_pred[:, :, i * image_size:h_max, j * image_size:w_max] = masks_pred_single

            masks_pred_softmax_batch = F.softmax(masks_pred, dim=1).cpu().numpy()
            masks_soft_batch = masks_pred_softmax_batch[:, 1:, :, :]
            masks_hard_batch = true_masks[:, 1:].cpu().numpy()

            masks_soft.extend(masks_soft_batch)
            masks_hard.extend(masks_hard_batch)

    masks_soft = np.array(masks_soft).transpose((1, 0, 2, 3))
    masks_hard = np.array(masks_hard).transpose((1, 0, 2, 3))
    masks_soft = np.reshape(masks_soft, (masks_soft.shape[0], -1))
    masks_hard = np.reshape(masks_hard, (masks_hard.shape[0], -1))
    # print(masks_hard[0].shape, masks_soft[0].shape)
    # print(masks_soft[0], masks_soft[0].min(), masks_soft[0].max(), type(masks_soft))
    # print(masks_hard[0], masks_hard[0].min(), masks_hard[0].max(), type(masks_hard))

    masks_hard[0] = masks_hard > 0.5
    ap = average_precision_score(masks_hard[0], masks_soft[0])

    auc = roc_auc_score(masks_hard[0], masks_soft[0])
    print(auc)
    return ap, auc


def denormalize(inputs):
    return (inputs * 255.).to(device=device, dtype=torch.uint8)


def generate_log_images(inputs_t, true_masks_t, masks_pred_softmax_t):
    true_masks = (true_masks_t * 255.).to(device=device, dtype=torch.uint8)
    masks_pred_softmax = (masks_pred_softmax_t.detach() * 255.).to(device=device, dtype=torch.uint8)
    inputs = denormalize(inputs_t)
    bs, _, h, w = inputs.shape
    pad_size = 5
    images_batch = (torch.ones((bs, 3, h, w * 3 + pad_size * 2)) * 255.).to(device=device, dtype=torch.uint8)

    images_batch[:, :, :, :w] = inputs

    images_batch[:, :, :, w + pad_size:w * 2 + pad_size] = 0
    images_batch[:, 0, :, w + pad_size:w * 2 + pad_size] = true_masks[:, 1, :, :]

    images_batch[:, :, :, w * 2 + pad_size * 2:] = 0
    images_batch[:, 0, :, w * 2 + pad_size * 2:] = masks_pred_softmax[:, 1, :, :]
    return images_batch


def image_to_patch(image, patch_size):
    bs, channel, h, w = image.shape
    return (image.reshape((bs, channel, h // patch_size, patch_size, w // patch_size, patch_size))
            .permute(2, 4, 0, 1, 3, 5)
            .reshape((-1, channel, patch_size, patch_size)))


def train_model(model, lesion, preprocess, train_loader, eval_loader, criterion, g_optimizer, g_scheduler,
                batch_size, num_epochs=5, start_epoch=0, start_step=0):
    model.to(device=device)
    tot_step_count = start_step

    best_ap = 0.
    dir_checkpoint = 'results/models_' + dataset_name + '_' + tatl + '_' + lesion.lower()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print('Starting epoch {}/{}.\t\n'.format(epoch + 1, start_epoch + num_epochs))
        g_scheduler.step()
        model.train()

        for inputs, true_masks in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)
            masks_pred = model(inputs)
            masks_pred_transpose = masks_pred.permute(0, 2, 3, 1)
            masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])
            true_masks_indices = torch.argmax(true_masks, 1)
            true_masks_flat = true_masks_indices.reshape(-1)
            loss_ce = criterion(masks_pred_flat, true_masks_flat.long())

            # Save images

            ce_weight = 1.
            g_loss = loss_ce * ce_weight

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            tot_step_count += 1

        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)

        if (epoch + 1) % 20 == 0:
            eval_ap, eval_auc = eval_model(model, eval_loader)
            with open("ap_during_learning_" + dataset_name + '_' + tatl + '_' + lesion + preprocess + ".txt", 'a') as f:
                f.write("epoch: " + str(epoch))
                f.write(" ap: " + str(eval_ap))
                f.write(" auc: " + str(eval_auc))
                f.write("\n")

            if eval_ap > best_ap:
                best_ap = eval_ap

                state = {
                    'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    'optimizer': g_optimizer.state_dict()
                }

                torch.save(state, \
                           os.path.join(dir_checkpoint, 'model_' + preprocess + '.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--preprocess', type=str, default='7')
    parser.add_argument('--lesion', type=str, default='EX')
    args = parser.parse_args()
    # Set random seed for Pytorch and Numpy for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Unet Model loaded...")
    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your dataset)
    )
    # print(model)
    # model = UNet(n_channels=3, n_classes=2)
    g_optimizer = optim.SGD(model.parameters(),
                            lr=config.G_LEARNING_RATE,
                            momentum=0.9,
                            weight_decay=0.0005)
    resume = False
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch'] + 1
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            g_optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model loaded from {}'.format(resume))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    else:
        start_epoch = 0
        start_step = 0

    train_image_paths, train_mask_paths = get_images_fgadr_from_pd(image_dir, args.preprocess, phase='train')
    eval_image_paths, eval_mask_paths = get_images_fgadr_from_pd(image_dir, args.preprocess, phase='eval')
    print(len(train_image_paths), len(eval_image_paths))

    train_dataset = FGADRDataset(train_image_paths, train_mask_paths, config.LESION_IDS[args.lesion], transform=
    Compose([
        RandomRotation(rotation_angle),
        RandomCrop(image_size),
    ]))
    eval_dataset = FGADRDataset(eval_image_paths, eval_mask_paths, config.LESION_IDS[args.lesion])

    train_loader = DataLoader(train_dataset, batchsize, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batchsize, shuffle=False)

    for data in eval_loader:
        img, mask = data
        print("img: ", img.shape, img.min(), img.max())
        print("Mask: ", mask.shape, mask.min(), mask.max())
    # exit()

    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=200, gamma=0.9)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(config.CROSSENTROPY_WEIGHTS).to(device))

    train_model(model, args.lesion, args.preprocess, train_loader, eval_loader, criterion, g_optimizer, g_scheduler, \
                batchsize, num_epochs=config.EPOCHES, start_epoch=start_epoch, start_step=start_step)

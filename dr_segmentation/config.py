#!/usr/bin/env python
# -*- coding: utf-8 -*-

LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3, 'SG':4}

#Modify the general parameters.
IMAGE_DIR = '../HEDNet_cGAN/data'
NET_NAME = 'unet'
IMAGE_SIZE = 512

#Modify the parameters for training.
# EPOCHES = 5000
EPOCHES = 2500
TRAIN_BATCH_SIZE = 4
G_LEARNING_RATE = 0.001
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = None
LOAD_PRETRAINED = '/home/haal02/projects/DR-segmentation/UNet/results/new_models_sg/model_7.pth.tar'

#!/usr/bin/env python
# -*- coding: utf-8 -*-
LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3, 'SG':4}
DATASET_NAME = 'FGADR'
TATL = 'NO_TATL' #'TATL'
#Modify the general parameters.
IMAGE_DIR = '/home/hnguyen/DATA/SSL-Medical/KOTORI/Eye_Dataset/FGADR/Seg-set'
NET_NAME = 'unet'
IMAGE_SIZE = 512

#Modify the parameters for training.
EPOCHES = 1500
TRAIN_BATCH_SIZE = 4
G_LEARNING_RATE = 0.001
LESION_DICE_WEIGHT = 0.
ROTATION_ANGEL = 20
CROSSENTROPY_WEIGHTS = [0.1, 1.]
RESUME_MODEL = None
LOAD_PRETRAINED = None
# RESUME_MODEL = 'results/models_FGADR_NO_TATL_ma/model_1_bkc1.pth.tar'

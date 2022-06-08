#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
from preprocess import clahe_gridsize
import cv2
import pandas as pd
import torch.nn as nn

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

train_ratio = 0.8
eval_ratio = 0.2

def get_images(image_dir, preprocess='0', phase='train'):
    if phase == 'train' or phase == 'eval':
        setname = 'TrainingSet'
    elif phase == 'test':
        setname = 'TestingSet' 

    limit = 2
    grid_size = 8
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess))
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname))
        
        # compute mean brightess
        meanbright = 0.
        images_number = 0
        for tempsetname in ['TrainingSet', 'TestingSet']:
            imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/'+ tempsetname + '/*.jpg'))
            imgs_ori.sort()
            images_number += len(imgs_ori)
            # mean brightness.
            for img_path in imgs_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'Groundtruths', tempsetname, 'Mask', img_name + '_MASK.tif')
                gray = cv2.imread(img_path, 0)
                mask_img = cv2.imread(mask_path, 0)
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
        meanbright /= images_number
        
        imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/' + setname + '/*.jpg'))
        
        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None], '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright], '6': [True, True, None], '7': [True, True, meanbright]}
        for img_path in imgs_ori:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mask_path = os.path.join(image_dir, 'Groundtruths', setname, 'Mask', img_name + '_MASK.tif')
            clahe_img = clahe_gridsize(img_path, mask_path, denoise=preprocess_dict[preprocess][0], contrastenhancement=preprocess_dict[preprocess][1], brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit, gridsize=grid_size)
            cv2.imwrite(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname, os.path.split(img_path)[-1]), clahe_img)
        
    imgs = glob.glob(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname, '*.jpg'))

    imgs.sort()
    mask_paths = []
    train_number = int(len(imgs) * train_ratio)
    eval_number = int(len(imgs) * eval_ratio)
    if phase == 'train':
        image_paths = imgs[:train_number]
    elif phase == 'eval':
        image_paths = imgs[train_number:]
    else:
        image_paths = imgs
    mask_path = os.path.join(image_dir, 'Groundtruths', setname)
    lesions = ['HardExudates', 'Haemorrhages', 'Microaneurysms', 'SoftExudates', 'Full_Segmentation', 'Mask']
    lesion_abbvs = ['EX', 'HE', 'MA', 'SE', 'SG', 'MASK']
    for image_path in image_paths:
        paths = []
        name = os.path.split(image_path)[1].split('.')[0]
        for lesion, lesion_abbv in zip(lesions, lesion_abbvs):
            candidate_path = os.path.join(mask_path, lesion, name + '_' + lesion_abbv + '.tif')
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                paths.append(None)
        mask_paths.append(paths)
    return image_paths, mask_paths


def get_images_fgadr(image_dir, preprocess='0', phase='train'):
    if phase == 'train' or phase == 'eval':
        setname = 'TrainingSet'
    elif phase == 'test':
        setname = 'TestingSet'

    limit = 2
    grid_size = 8
    # image_dir = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set'
    clahe = 'Clahe'
    # if not os.path.exists(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess)):
    #     os.mkdir(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess))

    if not os.path.exists(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess, setname))

        # compute mean brightess
        meanbright = 0.
        images_number = 0

        imgs_ori = glob.glob(os.path.join(image_dir, 'Original_Images' + '/*.png'))
        print("coming in imgs")
        imgs_ori.sort()
        images_number += len(imgs_ori)
        # mean brightness.
        for img_path in imgs_ori:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mask_path = os.path.join(image_dir, 'Mask', img_name + '_MASK.png')
            gray = cv2.imread(img_path, 0)
            mask_img = cv2.imread(mask_path, 0)
            brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
            meanbright += brightness
        meanbright /= images_number

        imgs_ori = glob.glob(os.path.join(image_dir, 'Original_Images/' + '/*.png'))
        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None],
                           '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright],
                           '6': [True, True, None], '7': [True, True, meanbright]}


        for img_path in imgs_ori:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mask_path = os.path.join(image_dir, 'Mask', img_name + '_MASK.png')
            clahe_img = clahe_gridsize(img_path, mask_path, denoise=preprocess_dict[preprocess][0],
                                       contrastenhancement=preprocess_dict[preprocess][1],
                                       brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit,
                                       gridsize=grid_size)
            cv2.imwrite(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess, os.path.split(img_path)[-1]),
                        clahe_img)


    imgs = glob.glob(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess, '*.png'))

    imgs.sort()
    mask_paths = []
    img_train_ratio = 0.6
    img_eval_ratio = 0.2
    img_test_ratio =0.2
    # train_number = int(len(imgs) * train_ratio)
    # eval_number = int(len(imgs) * eval_ratio)
    train_number = 43
    eval_number = 11
    if phase == 'train':
        image_paths = imgs[:150]
    elif phase == 'eval':
        image_paths = imgs[1500:1510]
    else:
        image_paths = imgs
    mask_path = image_dir
    lesions = ['HardExudate_Masks', 'Hemohedge_Masks', 'Microaneurysms_Masks', 'SoftExudate_Masks', 'Full_Segmentation', 'Mask']
    lesion_abbvs = ['EX', 'HE', 'MA', 'SE', 'SG', 'MASK']
    for image_path in image_paths:
        paths = []
        name = os.path.split(image_path)[1].split('.')[0]
        for lesion in lesions:
            if lesion == 'Mask':
                candidate_path = os.path.join(mask_path, lesion, name + '_' + 'MASK' + '.png')
            else:
                candidate_path = os.path.join(mask_path, lesion, name+'.png')
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                # print("No path")
                paths.append(None)

        mask_paths.append(paths)
    return image_paths, mask_paths


def get_images_fgadr_from_pd(image_dir, preprocess='0', phase='train'):

    if phase == 'train' or phase == 'eval':
        setname = 'TrainingSet'
    elif phase == 'test':
        setname = 'TestingSet'

    limit = 2
    grid_size = 8
    # image_dir = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set'
    # '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set'
    clahe = 'Clahe'
    fgadr_df = pd.read_csv(os.path.join(image_dir, 'DR_Seg_Grading_Label_Filtered.csv'))
    # fgadr_df = pd.read_csv(os.path.join(image_dir, 'DR_Seg_Grading_Label_Combined.csv'))
    # imgs = glob.glob(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess, '*.png'))
    imgs = list(fgadr_df['clahe0'])

    # imgs.sort()
    mask_paths = []
    img_train_ratio = 0.6
    img_eval_ratio = 0.2
    img_test_ratio = 0.2
    train_number = int(len(imgs) * img_train_ratio)
    eval_number = int(len(imgs) * img_eval_ratio)
    test_number = int(len(imgs) * img_test_ratio)
    eval_split = len(imgs) - test_number
    # train_number = 43
    # eval_number = 11
    if phase == 'train':
        image_paths = imgs[:train_number]
    elif phase == 'eval':
        image_paths = imgs[train_number:eval_split]
    else:
        image_paths = imgs
    mask_path = image_dir
    lesions = ['HardExudate_Masks', 'Hemohedge_Masks', 'Microaneurysms_Masks', 'SoftExudate_Masks', 'Full_Segmentation',
               'Mask']
    lesion_abbvs = ['EX', 'HE', 'MA', 'SE', 'SG', 'MASK']
    for image_path in image_paths:
        paths = []
        name = os.path.split(image_path)[1].split('.')[0]
        for lesion in lesions:
            if lesion == 'Mask':
                candidate_path = os.path.join(mask_path, lesion, name + '_' + 'MASK' + '.png')
            else:
                candidate_path = os.path.join(mask_path, lesion, name + '.png')
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                # print("No path")
                paths.append(None)

        mask_paths.append(paths)
    return image_paths, mask_paths


def get_images_eyepacs(image_dir, preprocess='0', phase='train'):
    if phase == 'train' or phase == 'eval':
        setname = 'TrainingSet'
    elif phase == 'test':
        setname = 'TestingSet'

    limit = 2
    grid_size = 8
    image_dir = '/mnt/sda/haal02-data/Kaggle-Retinopathy-Datasets/dr-data'
    clahe = 'Clahe'
    if not os.path.exists(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess))

    # if not os.path.exists(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess)):
    #     os.mkdir(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess, setname))

        # compute mean brightess
        meanbright = 0.
        images_number = 0

        # reverse here
        # read blackmask folder then image

        imgs_mask = glob.glob(os.path.join(image_dir, 'train_mask' + '/*.jpeg'))

        print("coming in imgs ")
        imgs_mask.sort()
        images_number += len(imgs_mask)
        # mean brightness.
        for img_path in imgs_mask:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            orig_img_path = os.path.join(image_dir, 'train', img_name + '.jpeg')
            mask_path = img_path
            gray = cv2.imread(orig_img_path, 0)
            mask_img = cv2.imread(mask_path, 0)
            brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
            meanbright += brightness
        meanbright /= images_number


        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None],
                           '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright],
                           '6': [True, True, None], '7': [True, True, meanbright]}


        for img_path in imgs_mask:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            orig_img_path = os.path.join(image_dir, 'train', img_name + '.jpeg')
            mask_path = img_path
            clahe_img = clahe_gridsize(img_path, mask_path, denoise=preprocess_dict[preprocess][0],
                                       contrastenhancement=preprocess_dict[preprocess][1],
                                       brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit,
                                       gridsize=grid_size)
            cv2.imwrite(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess, os.path.split(img_path)[-1]),
                        clahe_img)


    imgs = glob.glob(os.path.join(image_dir, clahe, 'Images_CLAHE' + preprocess, '*.jpeg'))

    # imgs.sort()
    mask_paths = []

    mask_path = image_dir
    lesions = ['HardExudate_Masks', 'Hemohedge_Masks', 'Microaneurysms_Masks', 'SoftExudate_Masks', 'SoftExudate_Masks', 'Mask']
    lesion_abbvs = ['EX', 'HE', 'MA', 'SE', 'SG', 'MASK']
    for image_path in imgs:
        name = os.path.split(image_path)[1].split('.')[0]
        black_mask_path = os.path.join(image_dir, 'train_mask', name + '.jpeg')
        paths = [None, None, None, None, None, black_mask_path]
        mask_paths.append(paths)

    return imgs[0:300], mask_paths[0:300]
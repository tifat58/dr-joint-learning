#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from config import IMAGE_SIZE  # or set IMAGE_SIZE manually if not in config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_prediction(pred_mask, save_path):
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(pred_mask).save(save_path)

def main(args):
    image = load_image(args.image_path).to(device)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    )

    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from {args.model}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.model}")

    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        pred_softmax = F.softmax(output, dim=1)
        pred_class = torch.argmax(pred_softmax, dim=1)  # [B, H, W]

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, 'prediction.png')
    save_prediction(pred_class, pred_path)
    print(f"Saved prediction to {pred_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save predictions')
    args = parser.parse_args()
    main(args)

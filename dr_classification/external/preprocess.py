# this file built on solution from kaggle team o_O: https://github.com/sveitser/kaggle_diabetic
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
from multiprocessing import Process
import cv2


# src = '/mnt/sda/haal02-data/IDRID-updated/DiseaseGrading/OriginalImages/TrainingSet'
# tgt = '/mnt/sda/haal02-data/IDRID-updated/Processed/DiseaseGrading/OriginalImages/TrainingSet'

src = '/mnt/sda/haal02-data/IDRID-updated/DiseaseGrading/OriginalImages/TestingSet'
tgt = '/mnt/sda/haal02-data/IDRID-updated/Processed/DiseaseGrading/OriginalImages/TestingSet'


def main():
    jobs = []
    for root, _, imgs in os.walk(src):
        for img in tqdm(imgs):
            src_path = os.path.join(root, img)
            tgt_dir = root.replace(src, tgt)
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            tgt_path = os.path.join(tgt_dir, img)
            print(tgt_path)
            image = cv2.imread(src_path)
            scale = 512
            a, r, s = scaleRadius(image, scale)
            # a = image
            # #s u b t r a c t l o c a l mean c o l o r
            a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
            im = Image.fromarray(a)
            save(im, tgt_path)


def scaleRadius(img, scale) :
    x=img[int(img.shape[0]/2),:,:].sum(1)
    print(x)
    r=(x>x.mean()/10).sum()/2
    s=scale * 1.0 / r
    print(r, s)
    return cv2.resize(img,(0,0), fx=s, fy=s), r, s




def save(img, fname):
    img.save(fname, quality=100, subsampling=0)


if __name__ == "__main__":
    main()
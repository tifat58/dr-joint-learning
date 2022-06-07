from torchvision import datasets
from torch.utils.data import Dataset
import os

from .loader import pil_loader


class CustomizedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        super(CustomizedImageFolder, self).__init__(root, transform, target_transform, loader=loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None, loader=pil_loader):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs[0:830]
        self.loader = loader
        self.transform = transform
        self.targets = [img[1] for img in imgs]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        for_grad = False
        img_path, label = self.imgs[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if for_grad:
            # name = img_path.split('/')[-1]
            # seg_path_root = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/Full_Segmentation/'
            # mask_path = os.path.join(seg_path_root, name)

            return img, label, img_path
        else:
            return img, label

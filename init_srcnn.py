import numpy as np
import os
from PIL import Image, ImageOps, ImageFile
import random
import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import sys
sys.setrecursionlimit(10000)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class PairedRandomFlipRotate:
    """对两张图像同时进行随机水平/垂直翻转和旋转，支持 Compose 使用"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        # 水平翻转
        if random.random() < self.p:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        # 垂直翻转
        if random.random() < self.p:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        # 随机旋转
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)

        return img1, img2


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_crop_img(img, patch_size):
    if (img.size[0] < patch_size):
        img = ImageOps.crop(img, border=img.size[0] - patch_size)
    if (img.size[1] < patch_size):
        img = ImageOps.crop(img, border=img.size[1] - patch_size)
    sp1, sp2 = random.randint(0, img.size[0] - patch_size), random.randint(0, img.size[1] - patch_size)
    return img.crop((sp1, sp2, sp1 + patch_size, sp2 + patch_size))

def make_LR_patch(img, upscale_factor, withupsampling):
    H, W = img.size
    if withupsampling == True:
        img0 = img.resize((H//upscale_factor, W//upscale_factor), Image.Resampling.BICUBIC)
        return img0.resize((H, W), Image.Resampling.BICUBIC)
    else:
        return img.resize((H//upscale_factor, W//upscale_factor), Image.Resampling.BICUBIC)

def data_initialization(img):
    return np.expand_dims(np.asarray(img, np.float32), axis=0) / 255.

def get_all_tif_files(directory):
    tif_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_files.append(os.path.join(root, file))
    return tif_files

def get_all_png_files(directory):
    png_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files

def get_all_jpg_files(directory):
    jpg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

class TrainDataset(Dataset):
    def __init__(self, imgpath, patch_size=None, upscale_factor=None, upsampling=False):
        super(TrainDataset, self).__init__()
        self.HR_img_list = get_all_jpg_files(imgpath)
        self.upscale_factor = upscale_factor
        self.withupsampling = upsampling
        self.patch_size = patch_size
        self.augment = PairedRandomFlipRotate(p=0.5)

    def __getitem__(self, index):
        HR_img = Image.open(self.HR_img_list[index]).convert('L')

        sp1, sp2 = (
            random.randint(0, HR_img.size[0] - self.patch_size), random.randint(0, HR_img.size[1] - self.patch_size))

        HR_patch = HR_img.crop((sp1, sp2, sp1 + self.patch_size, sp2 + self.patch_size))
        LR_patch = make_LR_patch(HR_patch, self.upscale_factor, self.withupsampling)

        LR_patch, HR_patch = torch.as_tensor(data_initialization(LR_patch)), torch.as_tensor(data_initialization(HR_patch))

        return self.augment(LR_patch, HR_patch)

    def __len__(self):
        return len(self.HR_img_list)


class EvalDataset(Dataset):
    def __init__(self, imgpath, patch_size=None, upscale_factor=None, upsampling=False):
        super(EvalDataset, self).__init__()
        self.HR_img_list = get_all_jpg_files(imgpath)
        self.upscale_factor = upscale_factor
        self.withupsampling = upsampling
        self.patch_size = patch_size

    def __getitem__(self, index):
        HR_img = Image.open(self.HR_img_list[index]).convert('L')

        sp1, sp2 = (
            random.randint(0, HR_img.size[0] - self.patch_size), random.randint(0, HR_img.size[1] - self.patch_size))

        HR_patch = HR_img.crop((sp1, sp2, sp1 + self.patch_size, sp2 + self.patch_size))
        LR_patch = make_LR_patch(HR_patch, self.upscale_factor, self.withupsampling)

        return torch.as_tensor(data_initialization(LR_patch)), torch.as_tensor(data_initialization(HR_patch))

    def __len__(self):
        return len(self.HR_img_list)

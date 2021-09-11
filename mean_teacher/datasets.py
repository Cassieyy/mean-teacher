# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import torchvision.transforms as transforms
from . import data
from .utils import export
import torch.utils.data as Data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
import os


@export
# def imagenet():
def GoPro():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': '../datasets/GoPro'
        # 'datadir': 'data-local/images/ilsvrc2012/',
        # 'num_classes': 1000
    }

# --- Imports --- #


# --- Training dataset --- #
class TrainData(Data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename):
        super().__init__()
        # train_list = train_data_dir + train_filename
        train_list = os.path.join(train_data_dir, train_filename)
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('rain','norain') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/',input_name)[-1][:-4]

        input_img = Image.open(os.path.join(self.train_data_dir, 'blur_crops', input_name))

        try:
            gt_img = Image.open(os.path.join(self.train_data_dir, 'sharp_crops', gt_name))
        except:
            gt_img = Image.open(os.path.join(self.train_data_dir, 'sharp_crops', gt_name)).convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        ###
        train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
    ]))
        input_img = train_transformation(input_img)
        gt_img = 
        
        eval_transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        ###

        # --- Check the channel is 3 or not --- #
        if list(input_im.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return input_im, gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


# --- Validation/test dataset --- #
class ValData(Data.Dataset):
    def __init__(self, val_data_dir,val_filename):
        super().__init__()
        val_list = os.path.join(val_data_dir, val_filename)
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('rain','norain') for i in input_names] # 对于gopro的话 文件名都是相同的

        self.input_names = input_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        input_img = Image.open(os.path.join(self.val_data_dir, 'input', input_name))
        gt_img = Image.open(os.path.join(self.val_data_dir, 'target', gt_name))

        # Resizing image in the multiple of 16"
        wd_new,ht_new = input_img.size
        if ht_new>wd_new and ht_new>1024:
            wd_new = int(np.ceil(wd_new*1024/ht_new))
            ht_new = 1024
        elif ht_new<=wd_new and wd_new>1024:
            ht_new = int(np.ceil(ht_new*1024/wd_new))
            wd_new = 1024
        wd_new = int(16*np.ceil(wd_new/16.0))
        ht_new = int(16*np.ceil(ht_new/16.0))
        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)
        # print(input_im.shape, gt.shape, input_name)3, 576, 1024
        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)



@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }

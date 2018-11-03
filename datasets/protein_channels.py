import csv
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomCrop
)


def strong_aug(p=.5, config=None):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.001, scale_limit=0.1, rotate_limit=20, p=.2),
        Compose([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        RandomCrop(height=224, width=224, p=1.0),
    ])


def tta_aug(p=.5, config=None):
    return Compose([
        # RandomCrop(height=224, width=224, p=1.0),
        # HorizontalFlip(),
        # OneOf([
        #     IAAAdditiveGaussianNoise(),
        #     GaussNoise(),
        # ], p=0.2),
        # OneOf([
        #     MotionBlur(p=.2),
        #     MedianBlur(blur_limit=3, p=.1),
        #     Blur(blur_limit=3, p=.1),
        # ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.001, scale_limit=0.1, rotate_limit=10, p=.2),
        # Compose([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        # OneOf([
        #     CLAHE(clip_limit=2),
        #     IAASharpen(),
        #     IAAEmboss(),
        #     RandomContrast(),
        #     RandomBrightness(),
        # ], p=0.3),
        # HueSaturationValue(p=0.3),
    ])


class ProteinChannelsDataset(Dataset):
    def __init__(self, config, name):
        self.data_folder = os.path.join(config['data_loader']['data_dir'], name)
        self.name = name
        self.config = config
        self.labels = []
        self.images, self.targets = self.get_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_red = np.array(Image.open(self.images[index] + '_red.png'))  # .resize((224, 224)))
        image_green = np.array(Image.open(self.images[index] + '_green.png'))  # .resize((224, 224)))
        image_blue = np.array(Image.open(self.images[index] + '_blue.png'))  # .resize((224, 224)))
        image_yellow = np.array(Image.open(self.images[index] + '_yellow.png'))  # .resize((224, 224)))

        pseudo_image = np.stack((image_red, image_green, image_blue), axis=2)

        if self.name == 'train':
            augmentation = strong_aug(p=1.0, config=self.config)
        else:
            augmentation = tta_aug(p=1.0, config=self.config)

        data = {"image": pseudo_image,
                "mask": image_yellow}
        augmented = augmentation(**data)
        pseudo_image, mask = augmented["image"], augmented["mask"]

        mask = np.expand_dims(mask, axis=2)

        image = torch.cat((
            torch.from_numpy(pseudo_image).float(),
            torch.from_numpy(mask).float(),
        ), dim=2)

        image = torch.transpose(image, dim0=0, dim1=2)

        target = self.targets[index]

        target = torch.from_numpy(target).float()

        return image, target

    def get_images(self):
        if self.name == 'train':
            images, targets = self.get_images_from_csv(os.path.join(self.config['data_loader']['data_dir'],
                                                                    'train.csv'), 'train')
        else:
            images, targets = self.get_images_from_csv(os.path.join(self.config['data_loader']['data_dir'],
                                                                    'sample_submission.csv'), 'test')
        return images, targets

    def get_images_from_csv(self, path_to_csv, train_or_test):
        rare_classes = [27, 15, 10, 8, 20, 17, 24, 26, 16, 13, 12, 22, 18, 6, 14]
        images = []
        targets = []
        with open(path_to_csv, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', dialect='excel')
            rows = list(reader)
            for i, row in tqdm(enumerate(rows[1:])):
                image = os.path.join(self.config['data_loader']['data_dir'], train_or_test, row[0])
                images.append(image)
                targets_list = [int(i) for i in row[1].split(' ')]
                self.labels.append(targets_list)
                target = np.zeros(self.config['class_number'])
                target[targets_list] = 1.0
                targets.append(target)

                if set(targets_list).intersection(rare_classes):
                    for additional_image in range(1):
                        images.append(image)
                        targets.append(target)
        print('images ', len(images))
        print('targets ', len(targets))

        return images, targets

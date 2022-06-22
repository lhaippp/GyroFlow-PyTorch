import torch
import random
import numbers

import numpy as np
import imgaug.augmenters as iaa

from skimage.transform import resize as imresize
from torchvision import transforms
from PIL import ImageFilter

from . import stn


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = [img[y1: y1 + th, x1: x1 + tw] for img in inputs]
        return inputs


class RandomSwap(object):
    def __call__(self, inputs):
        if random.random() < 0.5:
            inputs = inputs[::-1]
        return inputs


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, inputs):
        if random.random() < 0.5:
            inputs = [np.copy(np.fliplr(im)) for im in inputs]
        return inputs


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert isinstance(array, np.ndarray)
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Zoom(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, image):
        h, w, _ = image.shape
        if h == self.new_h and w == self.new_w:
            return image
        image = imresize(image, (self.new_h, self.new_w))
        return image


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input):
        for t in self.co_transforms:
            input = t(input)
        return input


class ToPILImage(transforms.ToPILImage):
    def __call__(self, imgs):
        return [super(ToPILImage, self).__call__(im) for im in imgs]


class ColorJitter(transforms.ColorJitter):
    def __call__(self, imgs):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return [transform(im) for im in imgs]


class ToTensor(transforms.ToTensor):
    def __call__(self, imgs):
        return [super(ToTensor, self).__call__(im) for im in imgs]


class RandomGamma:
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, imgs):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return [self.adjust_gamma(im, gamma, self._clip_image) for im in imgs]


class RandomGaussianBlur:
    def __init__(self, p, max_k_sz):
        self.p = p
        self.max_k_sz = max_k_sz

    def __call__(self, imgs):
        if np.random.random() < self.p:
            radius = np.random.uniform(0, self.max_k_sz)
            imgs = [im.filter(ImageFilter.GaussianBlur(radius)) for im in imgs]
        return imgs


def homo_to_flow(homo, H=600, W=800):
    img_indices = stn.get_grid(batch_size=1, H=H, W=W, start=0)
    flow_gyro = stn.get_flow(homo, img_indices, image_size_h=H, image_size_w=W)
    return flow_gyro


def fetch_appearance_transform():
    transforms = [ToPILImage()]
    transforms.append(ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
    transforms.append(RandomGaussianBlur(0.5, 3))
    transforms.append(ToTensor())
    transforms.append(RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True))
    return transforms.Compose(transforms)


def fetch_input_transform(if_normalize=True):
    if if_normalize:
        transformer = transforms.Compose([ArrayToTensor(),
                                          transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])])
    else:
        transformer = transforms.Compose([ArrayToTensor()])
    return transformer


def fetch_spatial_transform(params):
    transforms = []
    if params.data_aug.crop:
        transforms.append(RandomCrop(params.data_aug.para_crop))
    if params.data_aug.hflip:
        transforms.append(RandomHorizontalFlip())
    if params.data_aug.swap:
        transforms.append(RandomSwap())
    return Compose(transforms)


def weather_transform():
    seq = iaa.Sequential([
        iaa.Fog(),
        iaa.Rain(drop_size=(0.10, 0.20))
    ])
    return seq

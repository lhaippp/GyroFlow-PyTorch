import torch
import random
import cv2

import numpy as np

class RandomCrop(object):
    def __init__(self, size=64):
        self.size = size

    def __call__(self, img):
        h, w, _ = img.shape
        th, tw = self.size, self.size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img[y1: y1 + th, x1: x1 + tw]
        return img

class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = np.copy(np.fliplr(img))
        return img

class ArrayToTensor(object):
    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array)
        return tensor.float()

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input

def fetch_transform():
    transforms = []
    transforms.append(RandomCrop())
    transforms.append(RandomHorizontalFlip())
    transforms.append(ArrayToTensor())
    return Compose(transforms)

if __name__ == '__main__':
    test = cv2.imread('/data/cd_brain/data/SIGNS/test_signs/0_IMG_5942.jpg')
    transforms = fetch_transform()
    output = transforms(test)
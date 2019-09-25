import torch
import numpy as np
import torchvision.transforms as transforms
import random
from autoaugment import ImageNetPolicy, CIFAR10Policy

_IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, num_crops=1, normalize=_IMAGENET_STATS):
    assert num_crops in [1, 5, 10], "num crops must be in {1,5,10}"
    convert_tensor = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(**normalize)])
    if num_crops == 1:
        t_list = [
            transforms.CenterCrop(input_size),
            convert_tensor
        ]
    else:
        if num_crops == 5:
            t_list = [transforms.FiveCrop(input_size)]
        elif num_crops == 10:
            t_list = [transforms.TenCrop(input_size)]
        # returns a 4D tensor
        t_list.append(transforms.Lambda(lambda crops:
                                        torch.stack([convert_tensor(crop) for crop in crops])))

    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def random_crop(input_size, scale_size=None, padding=None, normalize=_IMAGENET_STATS):
    scale_size = scale_size or input_size
    T = transforms.Compose([
        transforms.RandomCrop(scale_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])
    if input_size != scale_size:
        T.transforms.insert(1, transforms.Resize(input_size))
    return T


def cifar_autoaugment(input_size, scale_size=None, padding=None, normalize=_IMAGENET_STATS):
    scale_size = scale_size or input_size
    T = transforms.Compose([
        transforms.RandomCrop(scale_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(fillcolor=(128, 128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])
    if input_size != scale_size:
        T.transforms.insert(1, transforms.Resize(input_size))
    return T


def inception_preprocess(input_size, normalize=_IMAGENET_STATS):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def inception_autoaugment_preprocess(input_size, normalize=_IMAGENET_STATS):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(fillcolor=(128, 128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def inception_color_preprocess(input_size, normalize=_IMAGENET_STATS):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
        transforms.Normalize(**normalize)
    ])


def multi_transform(transform_fn, duplicates=1, dim=0):
    """preforms multiple transforms, useful to implement inference time augmentation or
     "batch augmentation" from https://openreview.net/forum?id=H1V4QhAqYQ&noteId=BylUSs_3Y7
    """
    if duplicates > 1:
        return transforms.Lambda(lambda x: torch.stack([transform_fn(x) for _ in range(duplicates)], dim=dim))
    else:
        return transform_fn


def get_transform(transform_name='imagenet', input_size=None, scale_size=None,
                  normalize=None, augment=True, cutout=None, autoaugment=False,
                  padding=None, duplicates=1, num_crops=1):
    normalize = normalize or _IMAGENET_STATS
    transform_fn = None
    if 'imagenet' in transform_name:  # inception augmentation is default for imagenet
        input_size = input_size or 224
        scale_size = scale_size or int(input_size * 8/7)
        if augment:
            if autoaugment:
                transform_fn = inception_autoaugment_preprocess(input_size,
                                                                normalize=normalize)
            else:
                transform_fn = inception_preprocess(input_size,
                                                    normalize=normalize)
        else:
            transform_fn = scale_crop(input_size=input_size, scale_size=scale_size,
                                      num_crops=num_crops, normalize=normalize)
    elif 'cifar' in transform_name:  # resnet augmentation is default for imagenet
        input_size = input_size or 32
        if augment:
            scale_size = scale_size or 32
            padding = padding or 4
            if autoaugment:
                transform_fn = cifar_autoaugment(input_size, scale_size=scale_size,
                                                 padding=padding, normalize=normalize)
            else:
                transform_fn = random_crop(input_size, scale_size=scale_size,
                                           padding=padding, normalize=normalize)
        else:
            scale_size = scale_size or 32
            transform_fn = scale_crop(input_size=input_size, scale_size=scale_size,
                                      num_crops=num_crops, normalize=normalize)
    elif transform_name == 'mnist':
        normalize = {'mean': [0.5], 'std': [0.5]}
        input_size = input_size or 28
        if augment:
            scale_size = scale_size or 32
            transform_fn = pad_random_crop(input_size, scale_size=scale_size,
                                           normalize=normalize)
        else:
            scale_size = scale_size or 32
            transform_fn = scale_crop(input_size=input_size, scale_size=scale_size,
                                      num_crops=num_crops, normalize=normalize)
    if cutout is not None:
        transform_fn.transforms.append(Cutout(**cutout))
    return multi_transform(transform_fn, duplicates)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    taken from https://github.com/uoguelph-mlrg/Cutout


    Args:
        holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, holes, length):
        self.holes = holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

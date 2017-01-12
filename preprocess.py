import torch
import torchvision.transforms as transforms

_default_norm = transforms.Normalize(mean=3 * [0.5], std=3 * [0.5])
_imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=_default_norm):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=_default_norm):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=_default_norm):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def inception_preproccess(input_size, normalize=_imagenet_norm):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
        normalize,
    ])


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = normalize or _imagenet_norm
    if name == 'imagenet':
        scale_size = scale_size or 256
        input_size = input_size or 224
        if augment:
            return inception_preproccess(input_size, normalize=normalize)
        else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)
    elif 'cifar' in name:
        input_size = input_size or 32
        if augment:
            scale_size = scale_size or 40
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize)
        else:
            scale_size = scale_size or 32
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = torch.Tensor(3).normal_(0, self.alphastd)
        rgb = self.eigvec.clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        img.add_(rgb.view(3, 1, 1).expand_as(img))
        return img


class Grayscale(object):

    def __call__(self, img):
        img[0].mul_(0.299).add_(0.587, img[1]).add_(0.114, img[2])
        img[1].copy_(img[0])
        img[2].copy_(img[0])
        return img


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img.clone())
        alpha = 1.0 + torch.torch.Tensor(1).uniform_(-self.var, self.var)[0]
        img.lerp_(gs, alpha)

        return img


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = 1.0 + torch.torch.Tensor(1).uniform_(-self.var, self.var)[0]
        img.lerp_(gs, alpha)

        return img


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img.clone())
        gs.fill_(gs.mean())
        alpha = 1.0 + torch.torch.Tensor(1).uniform_(-self.var, self.var)[0]
        img.lerp_(gs, alpha)
        return img


class RandomOrder(object):
    """ Composes several transforms together in random order.
    For example:
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

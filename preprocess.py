import torchvision.transforms as transforms

_default_norm = transforms.Normalize(mean=3 * [0.5], std=3 * [0.5])
_imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])


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

import torch
from torchvision import transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def getTransform(
        mean,
        std,
        resize_size=224,
        crop_size=224,
        is_train=True
):
    if is_train:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            AddGaussianNoise(0.0, 0.01),
        ])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def getAllTransforms(
        resize_size=224,
        crop_size=224
):
    """
    Return transforms in the same view order used by all trainers:

        D -> 0
        C -> 1
        N -> 2
        P -> 3
    """
    mean_D = [0.3920, 0.3308, 0.2738]
    std_D = [0.3140, 0.2747, 0.2546]

    mean_C = [0.2646, 0.2010, 0.1370]
    std_C = [0.2935, 0.2254, 0.1877]

    mean_N = [0.3823, 0.3360, 0.2885]
    std_N = [0.3014, 0.2734, 0.2575]

    mean_P = [0.3537, 0.3030, 0.2510]
    std_P = [0.3044, 0.2691, 0.2507]

    transform_D1 = getTransform(
        mean_D,
        std_D,
        resize_size,
        crop_size,
        is_train=True
    )

    transform_C1 = getTransform(
        mean_C,
        std_C,
        resize_size,
        crop_size,
        is_train=True
    )
    transform_N1 = getTransform(
        mean_N,
        std_N,
        resize_size,
        crop_size,
        is_train=True
    )
    transform_P1 = getTransform(
        mean_P,
        std_P,
        resize_size,
        crop_size,
        is_train=True
    )

    transform_D2 = getTransform(
        mean_D,
        std_D,
        resize_size,
        crop_size,
        is_train=False
    )
    transform_C2 = getTransform(
        mean_C,
        std_C,
        resize_size,
        crop_size,
        is_train=False
    )
    transform_N2 = getTransform(
        mean_N,
        std_N,
        resize_size,
        crop_size,
        is_train=False
    )
    transform_P2 = getTransform(
        mean_P,
        std_P,
        resize_size,
        crop_size,
        is_train=False
    )

    transform_1 = [
        transform_D1,
        transform_C1,
        transform_N1,
        transform_P1,
    ]

    transform_2 = [
        transform_D2,
        transform_C2,
        transform_N2,
        transform_P2,
    ]

    return transform_1, transform_2
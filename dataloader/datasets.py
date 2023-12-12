import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF


class CustomDataset(Dataset):
    def __init__(self, base_dir, modality, split, transforms=None):
        super(CustomDataset, self).__init__()
        self.base_dir = base_dir
        self.modality = modality
        self.split = split
        self.transforms = transforms

        self.image_dir = os.path.join(self.base_dir, self.modality, 'images', self.split)
        self.label_dir = os.path.join(self.base_dir, self.modality, 'annotations', self.split)
        self.file_names = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_name = self.file_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name)

        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')

        # 将图像转换为张量
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        # 归一化图像
        image = TF.normalize(image, [0.5], [0.5])

        if self.split == 'test' or self.split == 'validation':
            # 在validation或test分割时二值化标签
            label = (label * 255).long()

        sample = {'image': image, 'label': label}

        if self.transforms:
            sample = self.transforms(sample)

        if self.split == 'test' or self.split == 'validation':
            return sample, image_name.replace('.png', '')
        else:
            return sample


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = torch.zeros((self.num_classes, label.shape[1], label.shape[2]), dtype=torch.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :] = (label[0] == i).type(torch.float32)
        sample['onehot_label'] = onehot_label
        return sample


class ToTensor(object):
    def __call__(self, sample):
        return {'image': sample['image'], 'label': sample['label'].long(), 'onehot_label': sample.get('onehot_label')}


class CenterCrop(object):
    """
    Center Crop 2D Slices
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='edge')
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='edge')

        (w, h) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop 2D Slices
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='edge')
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='edge')

        (w, h) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label, 'domain_label': sample['domain_label']}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1]), -2 * self.sigma, 2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = torch.zeros((self.num_classes, label.shape[1], label.shape[2]), dtype=torch.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :] = (label[0] == i).type(torch.float32)
        sample['onehot_label'] = onehot_label
        return sample


class ResizeToSize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 调整图像和标签大小
        image = TF.resize(image, self.output_size)
        label = TF.resize(label, self.output_size, interpolation=Image.NEAREST)

        return {'image': image, 'label': label}


import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import ToPILImage
class NormalizeTransform:
    """
    对MRI图像应用Z-Score标准化。
    """
    def __call__(self, image):
        mask = image > 0
        masked_image = image[mask]
        mean, std = masked_image.mean(), masked_image.std()
        normalized_image = np.where(mask, (image - mean) / std, 0)
        return normalized_image.copy()  # 确保返回一个新的数组

class RandomFlipTransform:
    """
    随机翻转图像。
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if np.random.rand() < self.p:
            flipped_image = np.flip(image, axis=0)  # 水平翻转
            return flipped_image.copy()  # 确保返回一个新的数组
        return image
# 结合多个转换
composed_transform = transforms.Compose([
    ToPILImage(),
    Resize((256, 256)),
    ToTensor(),  # 将numpy数组转换为torch.Tensor
    NormalizeTransform(),  # 应用Z-Score标准化
    RandomFlipTransform(p=0.5)  # 随机翻转图像
])
class BratsDataset(Dataset):
    """
    用于读取BraTS数据集的自定义数据集类。
    """
    def __init__(self, data_dir, transform=None, modality='t1'):
        self.data_dir = data_dir
        self.transform = transform
        self.modality = modality
        self.cases = [case for case in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, case))]
        self.all_slices = []
        for case in self.cases:
            case_path = os.path.join(self.data_dir, case)
            modality_file = f'{case}_{self.modality}.nii'
            label_file = f'{case}_seg.nii'
            modality_path = os.path.join(case_path, modality_file)
            label_path = os.path.join(case_path, label_file)
            modality_img = nib.load(modality_path).get_fdata(dtype=np.float32)
            label_img = nib.load(label_path).get_fdata(dtype=np.float32)
            num_slices = modality_img.shape[2]
            for slice_idx in range(num_slices):
                self.all_slices.append((case, slice_idx))

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, idx):
        case, slice_idx = self.all_slices[idx]
        case_path = os.path.join(self.data_dir, case)
        modality_file = f'{case}_{self.modality}.nii'
        label_file = f'{case}_seg.nii'
        modality_path = os.path.join(case_path, modality_file)
        label_path = os.path.join(case_path, label_file)
        modality_img = nib.load(modality_path).get_fdata(dtype=np.float32)
        label_img = nib.load(label_path).get_fdata(dtype=np.float32)
        modality_slice = modality_img[:, :, slice_idx]
        label_slice = label_img[:, :, slice_idx]

        if self.transform:
            modality_slice = self.transform(modality_slice)
            label_slice = self.transform(label_slice)  # 确保标签图像也经过相同的尺寸调整
        return modality_slice, label_slice

def get_loaders(data_dir, batch_size, modality='t1', val_size=0.1, test_size=0.1):
    dataset = BratsDataset(data_dir, transform=composed_transform, modality=modality)

    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_size, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (1 - test_size), random_state=42)

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

    return train_loader, val_loader, test_loader

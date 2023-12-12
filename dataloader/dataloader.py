import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
import pickle

class CIFAR10Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        # 加载data_batch_1到data_batch_5
        for i in range(1, 6):
            file_path = os.path.join(root, f'data_batch_{i}')
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            self.data.append(batch[b'data'])
            self.labels.append(batch[b'labels'])

        self.data = np.concatenate(self.data, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

def load_data(batch_size=32):
    # 定义图像转换流程
    transform = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小
        transforms.CenterCrop(224),  # 裁剪图像
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为伪RGB图像
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正则化
    ])

    # 加载 FashionMNIST 数据集作为测试集
    test_data = datasets.FashionMNIST(root='./Data/test/fashion-mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 加载 CIFAR-10 数据集作为训练集
    train_data = CIFAR10Dataset(root='/data/home/liumingsi/DDA2TTA/Data/train/cifar-10-batches-py', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

# 加载数据
#train_loader, test_loader = load_data()

# 测试：打印测试集的总批次数量
#print(f"测试集批次数量: {len(test_loader)}")

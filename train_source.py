import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dataloader.datasets import CustomDataset, ToTensor, CreateOnehotLabel, ResizeToSize
import torchvision.transforms as tfs
from model.eps_model import Unet as DiffusionUnet
from model.diffusion import DenoiseDiffusion
import segmentation_models_pytorch as smp
from torch.cuda.amp import GradScaler, autocast  # 引入AMP

# 定义 DiceLoss 类
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, score, target):
        target = target.float()
        smooth = 1e-5
        loss = 0
        for i in range(target.shape[1]):
            intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
            z_sum = torch.sum(score[:, i, ...])
            y_sum = torch.sum(target[:, i, ...])
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / target.shape[1]
        return loss

def train_unet(train_loader, model, criterion, optimizer, rank, scaler):
    model.train()
    total_loss = 0.0
    with tqdm(total=len(train_loader), desc="Training UNet", unit="batch") as tbar:
        for i, sample in enumerate(train_loader):
            images = sample['image']
            masks = sample['onehot_label']
            images, masks = images.to(rank), masks.to(rank)
            optimizer.zero_grad()

            with autocast():  # 使用 autocast
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()  # 使用 GradScaler
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            current_loss = total_loss / (i + 1)

            tbar.set_postfix(seg_loss=current_loss)
            tbar.update(1)

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def train_diffusion(train_loader, model, optimizer, rank, scaler):
    model.eps_model.train()
    total_loss = 0.0
    with tqdm(total=len(train_loader), desc="Training Diffusion Model", unit="batch") as tbar:
        for i, sample in enumerate(train_loader):
            image = sample['image']
            image = image.to(rank)
            optimizer.zero_grad()

            with autocast():  # 使用 autocast
                loss = model.loss(image)

            scaler.scale(loss).backward()  # 使用 GradScaler
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            current_loss = total_loss / (i + 1)

            tbar.set_postfix(diff_loss=current_loss)
            tbar.update(1)

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train UNet and Diffusion model for medical image segmentation')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--save_path', type=str, default='./checkpoints')
parser.add_argument('--log_path', type=str, default='./log')
parser.add_argument('--save_interval', type=int, default=1)
parser.add_argument('--root_dir', type=str, default='./Data')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--world_size', type=int, default=4, help='number of distributed processes')
parser.add_argument('--cuda_devices', type=str, default='6,7,8,9', help='CUDA devices (e.g., 0,1,2,3)')
args = parser.parse_args()

def setup(rank, world_size, cuda_devices):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def train(rank, args):
    setup(rank, args.world_size, args.cuda_devices)

    # 创建模型
    seg_model = smp.Unet(encoder_name="resnet50", in_channels=1, classes=args.n_classes).to(rank)
    seg_model = DDP(seg_model, device_ids=[rank])

    eps_model = DiffusionUnet(dim=256, init_dim=256, out_dim=1, dim_mults=(1, 2, 4, 8), channels=1).to(rank)
    eps_model = DDP(eps_model, device_ids=[rank])
    diff_model = DenoiseDiffusion(eps_model, n_steps=20, device=rank)

    criterion = DiceLoss().to(rank)
    seg_optimizer = optim.Adam(seg_model.parameters(), lr=args.lr)
    diff_optimizer = optim.Adam(eps_model.parameters(), lr=args.lr)

    scaler = GradScaler()  # 初始化 GradScaler

    # 创建数据加载器
    dataset = CustomDataset(base_dir=args.root_dir, modality='t2_pre', split='training',
                            transforms=tfs.Compose([ResizeToSize((256, 256)), CreateOnehotLabel(num_classes=args.n_classes), ToTensor()]))
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    diff_scheduler = torch.optim.lr_scheduler.OneCycleLR(diff_optimizer, max_lr=0.01, steps_per_epoch=len(train_loader),
                                                         epochs=args.epochs)
    seg_scheduler = torch.optim.lr_scheduler.OneCycleLR(seg_optimizer, max_lr=0.01, steps_per_epoch=len(train_loader),
                                                        epochs=args.epochs)

    for epoch in range(args.epochs):
        train_loss_diffusion = train_diffusion(train_loader, diff_model, diff_optimizer, rank, scaler)
        train_loss_unet = train_unet(train_loader, seg_model, criterion, seg_optimizer, rank, scaler)

        diff_scheduler.step()
        seg_scheduler.step()

        # 打印训练
        tqdm.write(
            f'Epoch {epoch + 1}/{args.epochs}: UNet Loss: {train_loss_unet:.4f}, Diffusion Loss: {train_loss_diffusion:.4f}')

        # 将训练和验证结果记录到日志
        with open(os.path.join(args.log_path, 'training_log.txt'), 'a') as log_file:
            log_file.write(
                f'Epoch {epoch + 1}: UNet Loss: {train_loss_unet:.4f}, Diffusion Loss: {train_loss_diffusion:.4f}\n')

        # 保存模型参数
        if (epoch + 1) % args.save_interval == 0:
            torch.save(seg_model.module.state_dict(), os.path.join(args.save_path, f'unet_epoch_{epoch + 1}.pth'))
            torch.save(diff_model.eps_model.module.state_dict(), os.path.join(args.save_path, f'diffusion_epoch_{epoch + 1}.pth'))
    cleanup()

if __name__ == '__main__':
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)

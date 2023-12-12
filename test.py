import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from dataloader.datasets import CustomDataset, ResizeToSize, ToTensor, CreateOnehotLabel
from model.eps_model import Unet as DiffusionUnet
from model.diffusion import DenoiseDiffusion
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred.argmax(dim=1).contiguous()
    intersection = (pred & target).sum(dim=(1, 2))
    union = (pred | target).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def test(test_loader, diffusion_model, seg_model, device):
    diffusion_model.eps_model.eval()
    seg_model.eval()
    total_dice = 0.0
    progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
    for sample, _ in progress_bar:
        image = sample['image']
        label = sample['onehot_label']
        image, label = image.to(device), label.to(device)
        image_g = diffusion_model.generate_refined_image(image, w=6, D=4)
        output = seg_model(image_g)

        dice_score = dice_coefficient(output, label.argmax(dim=1))
        total_dice += dice_score

    avg_dice = total_dice / len(test_loader)
    print(f"Average Dice Score: {avg_dice:.4f}")
    return avg_dice

def main():
    parser = argparse.ArgumentParser(description='Test on Flair')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--unet_model_path', type=str, default='./checkpoints/unet_epoch_2.pth')
    parser.add_argument('--eps_model_path', type=str, default='./checkpoints/diffusion_epoch_2.pth')
    parser.add_argument('--root_dir', type=str, default='./Data')
    parser.add_argument('--n_classes', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seg_model = smp.Unet(encoder_name="resnet50", in_channels=1, classes=args.n_classes).to(device)
    eps_model = DiffusionUnet(dim=256, init_dim=256, out_dim=256, dim_mults=(1, 2, 4, 8), channels=1).to(device)
    diff_model = DenoiseDiffusion(eps_model, n_steps=20, device=device)

    seg_model.load_state_dict(torch.load(args.unet_model_path, map_location=device))
    eps_model.load_state_dict(torch.load(args.eps_model_path, map_location=device))

    dataset = CustomDataset(base_dir=args.root_dir, modality='t2_pre', split='validation',
                            transforms=Compose([ResizeToSize((256, 256)), CreateOnehotLabel(num_classes=args.n_classes), ToTensor()]))
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)

    test(test_loader, diff_model, seg_model, device)

if __name__ == '__main__':
    main()

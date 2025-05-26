import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from DGSUNet import DGSUNet
from loss import structure_loss

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 注意这里是0，代表环境变量里第一个可见设备
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, 518, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    model = DGSUNet(args.dino_model_name, args.dino_hub_dir, args.sam_config_file, args.sam_ckpt_path)
    # model = DGSUNet(
    #     dino_model_name=args.dino_model_name,
    #     dino_hub_dir=args.dino_hub_dir,
    #     sam_config_file=args.sam_config_file,  # 确保参数名一致
    #     sam_ckpt_path=args.sam_ckpt_path
    # )
    model.to(device)
    optim = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):
        for i, batch in enumerate(dataloader):
            x1 = batch['image1']
            x2 = batch['image2']
            target = batch['label']
            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x2,x1)
            print(f"pred0 shape: {pred0.shape}, target shape: {target.shape}")  
            print(f"pred0 device: {pred0.device}, target device: {target.device}")  
            print(f"pred0 dtype: {pred0.dtype}, target dtype: {target.dtype}") 
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = 0.25*loss0 + 0.5*loss1 + loss2
            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))

        scheduler.step()
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'DGSUNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'DGSUNet-%d.pth' % (epoch + 1)))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.dino_model_name = "dinov2_vitl14"
    args.dino_hub_dir = "facebookresearch/dinov2"
    args.sam_config_file = "/data2/users/donghang/SAM2DINO-Seg/sam2_configs/sam2.1_hiera_l.yaml"
    args.sam_ckpt_path = "/data2/users/donghang/SAM2DINO-Seg/checkpoints/sam2.1_hiera_large.pt"
    args.train_image_path = "/data2/users/donghang/SAM2DINO-Seg/data/DUTS-TR/DUTS-TR-Image/"
    args.train_mask_path = "/data2/users/donghang/SAM2DINO-Seg/data/DUTS-TR/DUTS-TR-Mask/"
    args.save_path = "/data2/users/donghang/SAM2DINO-Seg/checkpoints"
    args.epoch = 50
    args.lr = 0.001
    args.batch_size = 4
    args.weight_decay = 5e-4
    main(args)

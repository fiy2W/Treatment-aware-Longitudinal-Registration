import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch._C import dtype
import torchvision.utils as vutils
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np

from dataloader.private_dataset import DatasetReg
import voxelmorph as vxm
from models.model_structure import KN_S


def train(
    net,
    net_sl,
    device,
    epochs=5,
    batch_size=1,
    lr=0.001,
    dir_checkpoint='',):

    train_data = DatasetReg(img_path='dataset_root/img_path', list_path='dataset_root/list_path/reg', mode='train')
    n_train = len(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    
    mi = vxm.torch.losses.MutualInformation()
    gauss = vxm.torch.losses.GaussKernel(gksize=7, gkstd=1, device=device)
    
    for epoch in range(epochs):
        net.train()
        net_sl.eval()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                if random.randint(0, 1):
                    fix = batch['fix'].to(device=device, dtype=torch.float32)
                    fix_seg = batch['fix_seg'].to(device=device, dtype=torch.float32)
                    mov = batch['mov'].to(device=device, dtype=torch.float32)
                    mov_seg = batch['mov_seg'].to(device=device, dtype=torch.float32)
                else:
                    fix = batch['mov'].to(device=device, dtype=torch.float32)
                    fix_seg = batch['mov_seg'].to(device=device, dtype=torch.float32)
                    mov = batch['fix'].to(device=device, dtype=torch.float32)
                    mov_seg = batch['fix_seg'].to(device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    d, w, h = fix.shape[2:5]
                    nd, nw, nh = 176, 176, 176
                    rd = random.randint(0, d-nd-1) if d>nd else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h//2-nh-1) if h//2>nh else 0
                    fix = fix[:,:,rd:rd+nd,rw:rw+nw,rh:rh+nh]
                    fix_seg = fix_seg[:,:,rd:rd+nd,rw:rw+nw,rh:rh+nh]
                    mov = mov[:,:,rd:rd+nd,rw:rw+nw,rh:rh+nh]
                    mov_seg = mov_seg[:,:,rd:rd+nd,rw:rw+nw,rh:rh+nh]

                    fix_seg = gauss(fix_seg)
                    mov_seg = gauss(mov_seg)

                    mov_sl_maps = net_sl.pose_encoder(mov[:,0:1])[-1][-1]
                    fix_sl_maps = net_sl.pose_encoder(fix[:,0:1])[-1][-1]

                    fix_seg = gauss(fix_seg)
                    mov_seg = gauss(mov_seg)
                
                reg_code = torch.rand((1,1), dtype=torch.float32, device=device)
                out = net(reg_code, mov, fix, with_seg=True, sourceseg=mov_seg)
                
                warp_sl_maps1 = net.transformer2(mov_sl_maps, out['flow_lr'])
                warp_sl_maps2 = net.transformer2(mov_sl_maps, out['flow_mr'])
                warp_sl_maps3 = net.transformer2(mov_sl_maps, out['flow_hr'])

                fix_down = F.interpolate(fix, scale_factor=0.5, mode='trilinear', align_corners=True)
                fix_seg_down = F.interpolate(fix_seg, scale_factor=0.5, mode='trilinear', align_corners=True)

                loss_similar = mi.loss(out['lr_img'][:,0:1,:,:,:], fix_down[:,0:1,:,:,:]) + \
                    mi.loss(out['mr_img'][:,0:1,:,:,:], fix_down[:,0:1,:,:,:]) + \
                    mi.loss(out['hr_img'][:,0:1,:,:,:], fix_down[:,0:1,:,:,:]) + \
                    mi.loss(out['lr_img_f'][:,0:1,:,:,:], fix[:,0:1,:,:,:]) + \
                    mi.loss(out['mr_img_f'][:,0:1,:,:,:], fix[:,0:1,:,:,:]) + \
                    mi.loss(out['hr_img_f'][:,0:1,:,:,:], fix[:,0:1,:,:,:]) + \
                    nn.SmoothL1Loss()(out['lr_img'][:,0:1,:,:,:], fix_down[:,0:1,:,:,:]) + \
                    nn.SmoothL1Loss()(out['mr_img'][:,0:1,:,:,:], fix_down[:,0:1,:,:,:]) + \
                    nn.SmoothL1Loss()(out['hr_img'][:,0:1,:,:,:], fix_down[:,0:1,:,:,:]) + \
                    nn.SmoothL1Loss()(out['lr_img_f'][:,0:1,:,:,:], fix[:,0:1,:,:,:]) + \
                    nn.SmoothL1Loss()(out['mr_img_f'][:,0:1,:,:,:], fix[:,0:1,:,:,:]) + \
                    nn.SmoothL1Loss()(out['hr_img_f'][:,0:1,:,:,:], fix[:,0:1,:,:,:])
                
                loss_sl = nn.SmoothL1Loss()(warp_sl_maps1, fix_sl_maps) + nn.SmoothL1Loss()(warp_sl_maps2, fix_sl_maps) + nn.SmoothL1Loss()(warp_sl_maps3, fix_sl_maps)
                
                loss_seg = nn.SmoothL1Loss()(out['lr_seg'], fix_seg_down) + vxm.torch.losses.DiceLoss()(out['lr_seg'], fix_seg_down) + \
                    nn.SmoothL1Loss()(out['mr_seg'], fix_seg_down) + vxm.torch.losses.DiceLoss()(out['mr_seg'], fix_seg_down) + \
                    nn.SmoothL1Loss()(out['hr_seg'], fix_seg_down) + vxm.torch.losses.DiceLoss()(out['hr_seg'], fix_seg_down)

                loss_grad = reg_code * vxm.torch.losses.Grad('l2', loss_mult=2).loss(None, out['flow_lr']) + \
                    reg_code * vxm.torch.losses.Grad('l2', loss_mult=2).loss(None, out['flow_mr']) + \
                    reg_code * vxm.torch.losses.Grad('l2', loss_mult=2).loss(None, out['flow_hr'])

                loss = 1 * loss_similar + 1 * loss_seg + 10 * loss_grad + 1 * loss_sl

                with torch.no_grad():
                    dice1 = (2*torch.sum(out['lr_seg']*fix_seg_down)/(torch.sum(out['lr_seg'])+torch.sum(fix_seg_down)))
                    dice2 = (2*torch.sum(out['mr_seg']*fix_seg_down)/(torch.sum(out['mr_seg'])+torch.sum(fix_seg_down)))
                    dice3 = (2*torch.sum(out['hr_seg']*fix_seg_down)/(torch.sum(out['hr_seg'])+torch.sum(fix_seg_down)))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(**{'mask': loss_seg.item(), 'similar': loss_similar.item(), 'grad': loss_grad.item(),
                                    'dsc_lr': dice1.item(), 'dsc_mr': dice2.item(), 'dsc_hr': dice3.item(),
                                    'code': reg_code[0,0].item(), 'sl': loss_sl.item()})

                pbar.update(fix.shape[0])

        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_{}.pth'.format(epoch+1)))
        logging.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train CPRN+SL',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-c', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-ks', '--kns', dest='kns', type=str, default=None,
                        help='Load KN-S model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')
    parser.add_argument('-s', '--save', dest='save', type=str, default='ckpt/cprn+sl',
                        help='save ckpt')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    dir_checkpoint = args.save
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    inshape = (176, 176, 176)

    net = vxm.torch.networks_conditional.VxmDense(inshape=inshape)
    net.to(device=device)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        print('[*] Load model from', args.load)
    
    net_sl = KN_S(3, in_channels=1, nb_channels=64, nb_maps=64, gauss_std=0.01, norm='gn')
    net_sl.to(device=device)
    if args.kns:
        net_sl.load_state_dict(torch.load(args.kns, map_location=device))
        print('[*] Load model from', args.kns)

    try:
        train(
            net=net,
            net_sl=net_sl,
            device=device,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            dir_checkpoint=dir_checkpoint,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
import argparse
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.nn.functional as F
import numpy as np

from dataloader.private_dataset import DatasetReg
import voxelmorph as vxm
from models.model_structure import KN_S
from models.model_abnormal import KN_A


def train(
    net,
    net_old,
    net_sl,
    net_al,
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

    reg_code_old = torch.from_numpy(np.array([0.5])).to(dtype=torch.float32, device=device).unsqueeze(0)
    for epoch in range(epochs):
        net.train()
        net_old.eval()
        net_sl.eval()
        net_al.eval()
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
                    
                    out = net_old(reg_code_old, fix, mov, with_seg=False)
                    warped = out['hr_img_f']
                    mov_al_maps = net_al.pose(warped[:,1:2], mov[:,1:2], mov_seg)[-1][-1]

                reg_code = torch.rand((1,1), dtype=torch.float32, device=device)
                out = net(reg_code, mov, fix, with_seg=True, sourceseg=mov_seg)

                warp_sl_maps1 = net.transformer2(mov_sl_maps, out['flow_lr'])
                warp_sl_maps2 = net.transformer2(mov_sl_maps, out['flow_mr'])
                warp_sl_maps3 = net.transformer2(mov_sl_maps, out['flow_hr'])
                
                mov_al_maps = torch.sum(mov_al_maps, dim=1, keepdim=True)
                warp_al_maps1 = net.transformer2(mov_al_maps, out['flow_lr'])
                warp_al_maps2 = net.transformer2(mov_al_maps, out['flow_mr'])
                warp_al_maps3 = net.transformer2(mov_al_maps, out['flow_hr'])
                
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
                
                loss_sl = nn.SmoothL1Loss()(warp_sl_maps1, fix_sl_maps) + \
                    nn.SmoothL1Loss()(warp_sl_maps2, fix_sl_maps) + \
                    nn.SmoothL1Loss()(warp_sl_maps3, fix_sl_maps)
                
                loss_seg = nn.SmoothL1Loss()(out['lr_seg'], fix_seg_down) + vxm.torch.losses.DiceLoss()(out['lr_seg'], fix_seg_down) + \
                    nn.SmoothL1Loss()(out['mr_seg'], fix_seg_down) + vxm.torch.losses.DiceLoss()(out['mr_seg'], fix_seg_down) + \
                    nn.SmoothL1Loss()(out['hr_seg'], fix_seg_down) + vxm.torch.losses.DiceLoss()(out['hr_seg'], fix_seg_down)
                
                loss_vp = nn.SmoothL1Loss()(torch.sum(warp_al_maps1), torch.sum(mov_al_maps)) + \
                    nn.SmoothL1Loss()(torch.sum(warp_al_maps2), torch.sum(mov_al_maps)) + \
                    nn.SmoothL1Loss()(torch.sum(warp_al_maps3), torch.sum(mov_al_maps))

                loss_grad = reg_code * vxm.torch.losses.Grad('l2', loss_mult=2).loss(None, out['flow_lr']) + \
                    reg_code * vxm.torch.losses.Grad('l2', loss_mult=2).loss(None, out['flow_mr']) + \
                    reg_code * vxm.torch.losses.Grad('l2', loss_mult=2).loss(None, out['flow_hr'])

                loss = 1 * loss_similar + 1 * loss_seg + 10 * loss_grad + 1 * loss_sl + 3e-5 * loss_vp

                with torch.no_grad():
                    dice1 = (2*torch.sum(out['lr_seg']*fix_seg_down)/(torch.sum(out['lr_seg'])+torch.sum(fix_seg_down)))
                    dice2 = (2*torch.sum(out['mr_seg']*fix_seg_down)/(torch.sum(out['mr_seg'])+torch.sum(fix_seg_down)))
                    dice3 = (2*torch.sum(out['hr_seg']*fix_seg_down)/(torch.sum(out['hr_seg'])+torch.sum(fix_seg_down)))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(**{'mask': loss_seg.item(), 'similar': loss_similar.item(), 'grad': loss_grad.item(),
                                    'dsc_lr': dice1.item(), 'dsc_mr': dice2.item(), 'dsc_hr': dice3.item(),
                                    'vp': loss_vp.item(),
                                    'code': reg_code[0,0].item(), 'sl': loss_sl.item()})

                pbar.update(fix.shape[0])

        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_{}.pth'.format(epoch+1)))
        logging.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train CPRN+SL+VP(AL)',
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
    parser.add_argument('-ka', '--kna', dest='kna', type=str, default=None,
                        help='Load KN-A model from a .pth file')
    parser.add_argument('-cprn', '--cprn', dest='cprn', type=str, default=None,
                        help='Load CPRN model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')
    parser.add_argument('-s', '--save', dest='save', type=str, default='ckpt/cprn+sl+vpal',
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
    
    net_old = vxm.torch.networks_conditional.VxmDense(inshape=inshape)
    net_old.to(device=device)
    if args.cprn:
        net_old.load_state_dict(torch.load(args.cprn, map_location=device))
        print('[*] Load model from', args.cprn)
    
    net_sl = KN_S(3, in_channels=1, nb_channels=64, nb_maps=64, gauss_std=0.01, norm='gn')
    net_sl.to(device=device)
    if args.kns:
        net_sl.load_state_dict(torch.load(args.kns, map_location=device))
        print('[*] Load model from', args.kns)

    net_al = KN_A(3, in_channels=1, nb_channels=64, nb_maps=1, gauss_std=0.25, norm='gn')
    net_al.to(device=device)
    if args.kna:
        net_al.load_state_dict(torch.load(args.kna, map_location=device))
        print('[*] Load model from', args.kna)

    try:
        train(
            net=net,
            net_old=net_old,
            net_sl=net_sl,
            net_al=net_al,
            device=device,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            dir_checkpoint=dir_checkpoint,
            inshape=inshape,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
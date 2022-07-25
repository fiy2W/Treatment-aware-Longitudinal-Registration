import argparse
import logging
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.nn.functional as F

import voxelmorph as vxm
from dataloader.private_dataset import DatasetReg
from models.model_abnormal import KN_A


def train(
    net_cprn,
    net,
    device,
    epochs=5,
    batch_size=1,
    lr=0.001,
    dir_checkpoint='',
    dir_visualization=''):

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

    vgg16 = models.vgg16(pretrained=True)
    vgg16.to(device=device)
    def vgg_features(x, net):
        out = []
        for i, layer in enumerate(net):
            x = layer(x)
            #print(i, x.shape, layer)
            if i in [4,9,16,23,30]:
                out.append(x)
                #print(x.shape)
        return out
    
    W_init = [100., 1.6, 2.3, 1.8, 2.8, 100.]

    reg_code = torch.from_numpy(np.array([0.5])).to(dtype=torch.float32, device=device).unsqueeze(0)

    for epoch in range(epochs):
        net_cprn.eval()
        net.train()
        vgg16.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:#total=n_train*n_train
            for batch in train_loader:
                if random.randint(0, 1):
                    fix = batch['fix'].to(device=device, dtype=torch.float32)
                    mov = batch['mov'].to(device=device, dtype=torch.float32)
                    fix_seg = batch['fix_seg'].to(device=device, dtype=torch.float32)
                else:
                    fix = batch['mov'].to(device=device, dtype=torch.float32)
                    mov = batch['fix'].to(device=device, dtype=torch.float32)
                    fix_seg = batch['mov_seg'].to(device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    d, w, h = fix.shape[2:5]
                    nd, nw, nh = 176, 176, 176
                    rd = random.randint(0, d-nd-1) if d>nd else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h//2-nh-1) if h//2>nh else 0
                    fix = fix[:,:,rd:rd+nd,rw:rw+nw,rh:rh+nh]
                    mov = mov[:,:,rd:rd+nd,rw:rw+nw,rh:rh+nh]
                    fix_seg = fix_seg[:,:,rd:rd+nd,rw:rw+nw,rh:rh+nh]

                with torch.no_grad():
                    out = net_cprn(reg_code, mov, fix, with_seg=False)
                    warped = out['hr_img_f'][:,1:2,:,:,:]
                    fix = fix[:,1:2,:,:,:]

                output = net(warped, fix, fix_seg)
                
                loss_recon = nn.L1Loss()(output['recon'], warped)
                loss_cross = nn.L1Loss()(output['cross'], fix)

                b,c,d,w,h = warped.shape
                bs = 2
                d_init = random.randint(0, output['recon'].shape[2]-bs-1)
                w_init = random.randint(0, output['recon'].shape[3]-bs-1)
                h_init = random.randint(0, output['recon'].shape[4]-bs-1)
                pred_input_d = torch.cat([
                    output['recon'][:,:,d_init:d_init+bs,:,:].permute((2,1,0,3,4)).reshape(bs*c,b,w,h).repeat(1,3,1,1),
                    output['cross'][:,:,d_init:d_init+bs,:,:].permute((2,1,0,3,4)).reshape(bs*c,b,w,h).repeat(1,3,1,1),], 0)
                real_input_d = torch.cat([
                    warped[:,:,d_init:d_init+bs,:,:].permute((2,1,0,3,4)).reshape(bs*c,b,w,h).repeat(1,3,1,1),
                    fix[:,:,d_init:d_init+bs,:,:].permute((2,1,0,3,4)).reshape(bs*c,b,w,h).repeat(1,3,1,1),], 0)
                pred_input_w = torch.cat([
                    output['recon'][:,:,:,w_init:w_init+bs,:].permute((3,1,0,2,4)).reshape(bs*c,b,d,h).repeat(1,3,1,1),
                    output['cross'][:,:,:,w_init:w_init+bs,:].permute((3,1,0,2,4)).reshape(bs*c,b,d,h).repeat(1,3,1,1),], 0)
                real_input_w = torch.cat([
                    warped[:,:,:,w_init:w_init+bs,:].permute((3,1,0,2,4)).reshape(bs*c,b,d,h).repeat(1,3,1,1),
                    fix[:,:,:,w_init:w_init+bs,:].permute((3,1,0,2,4)).reshape(bs*c,b,d,h).repeat(1,3,1,1),], 0)
                pred_input_h = torch.cat([
                    output['recon'][:,:,:,:,h_init:h_init+bs].permute((4,1,0,2,3)).reshape(bs*c,b,d,w).repeat(1,3,1,1),
                    output['cross'][:,:,:,:,h_init:h_init+bs].permute((4,1,0,2,3)).reshape(bs*c,b,d,w).repeat(1,3,1,1),], 0)
                real_input_h = torch.cat([
                    warped[:,:,:,:,h_init:h_init+bs].permute((4,1,0,2,3)).reshape(bs*c,b,d,w).repeat(1,3,1,1),
                    fix[:,:,:,:,h_init:h_init+bs].permute((4,1,0,2,3)).reshape(bs*c,b,d,w).repeat(1,3,1,1),], 0)
                
                pred_per_d = vgg_features(pred_input_d, vgg16.features)
                real_per_d = vgg_features(real_input_d, vgg16.features)
                pred_per_w = vgg_features(pred_input_w, vgg16.features)
                real_per_w = vgg_features(real_input_w, vgg16.features)
                pred_per_h = vgg_features(pred_input_h, vgg16.features)
                real_per_h = vgg_features(real_input_h, vgg16.features)
                
                rec_loss = 1 * loss_recon + 1 * loss_cross
                W_init[0] = W_init[0] + 0.01 * (rec_loss.item() - W_init[0])

                loss = rec_loss/W_init[0]*1000
                li = 1
                for ppd, rpd, ppw, rpw, pph, rph in zip(pred_per_d, real_per_d, pred_per_w, real_per_w, pred_per_h, real_per_h):
                    loss_per = torch.nn.MSELoss()(ppd, rpd.detach()) + torch.nn.MSELoss()(ppw, rpw.detach()) + torch.nn.MSELoss()(pph, rph.detach())
                    W_init[li] = W_init[li] + 0.01 * (loss_per.item() - W_init[li])
                    loss_per = loss_per/W_init[li]
                    loss = loss + loss_per*100
                    li += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(**{'recon': loss_recon.item(), 'cross': loss_cross.item()})

                pbar.update(fix.shape[0])
                
        
        with torch.no_grad():
            pts = torch.clamp(torch.max(output['gauss_maps'][-1], dim=1, keepdim=True)[0], 0, 1)
            pts = F.interpolate(pts, scale_factor=2, mode='trilinear', align_corners=True)
            f = F.interpolate(output['gauss_feat'], scale_factor=8, mode='trilinear', align_corners=True)
            d = warped.shape[2]
            save_img([
                [warped[0,0:1,d//2:d//2+1,:,:], fix[0,0:1,d//2:d//2+1,:,:], torch.max(warped[0,0:1,:,:,:], dim=1, keepdim=True)[0], torch.max(fix[0,0:1,:,:,:], dim=1, keepdim=True)[0]],
                [output['recon'][0,0:1,d//2:d//2+1,:,:], output['cross'][0,0:1,d//2:d//2+1,:,:], torch.max(output['recon'], dim=2, keepdim=True)[0][0,0:1,:,:,:], torch.max(output['cross'], dim=2, keepdim=True)[0][0,0:1,:,:,:]],
                [pts[0,0:1,d//2:d//2+1,:,:], pts[0,0:1,d//2:d//2+1,:,:], torch.clamp(torch.max(pts[0,0:1,:,:,:], dim=1, keepdim=True)[0],0,1), torch.clamp(f[0,:,d//2:d//2+1,:,:],0,1)],
                ], epoch, dir_visualization)
        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_{}.pth'.format(epoch+1)))
        logging.info(f'Checkpoint {epoch + 1} saved !')


def save_img(print_list, index, results_dir):
    # pdb.set_trace()
    nrow = len(print_list[0])
    cols = []
    for row in print_list:
        cols.append(torch.cat(row, dim=3))
    img = torch.cat(cols, dim=2)
    
    directory = results_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, '{:04d}'.format(index) + '.jpg')
    img = img.permute(1,0,2,3).contiguous()
    vutils.save_image(img.view(1,img.size(0),-1,img.size(3)).data, path, nrow=nrow, padding=0, normalize=True)


def get_args():
    parser = argparse.ArgumentParser(description='Train KeyNet-A model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-p', '--cprn', dest='cprn', type=str, default=None,
                        help='Load CPRN model from a .pth file')
    parser.add_argument('-c', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')
    parser.add_argument('-s', '--save', dest='save', type=str, default='ckpt/kn-a',
                        help='save ckpt')
    parser.add_argument('-v', '--visual', dest='visual', type=str, default='vis/kn-a',
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

    net_cprn = vxm.torch.networks_conditional.VxmDense(inshape=inshape)
    net_cprn.to(device=device)
    if args.cprn:
        net_cprn.load_state_dict(torch.load(args.cprn, map_location=device))
        print('[*] Load model from', args.cprn)
    
    net_kna = KN_A(3, in_channels=1, nb_channels=64, nb_maps=16, gauss_std=0.2, norm='gn')
    net_kna.to(device=device)
    if args.load:
        net_kna.load_state_dict(torch.load(args.load, map_location=device))
        print('[*] Load model from', args.load)
    
    try:
        train(
            net_cprn=net_cprn,
            net=net_kna,
            device=device,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            dir_checkpoint=dir_checkpoint,
            dir_visualization=args.visual,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
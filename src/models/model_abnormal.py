import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.layers import ConvBlock


class KN_A(nn.Module):
    def __init__(
        self,
        ndims,
        in_channels=1,
        nb_channels=32,
        nb_maps=10,
        gauss_std=0.1,
        norm='gn',
    ):
        super().__init__()
        
        self.image_encoder = Encoder(ndims, in_channels, nb_channels, norm=norm)
        self.pose_encoder = KeyPointer(ndims, in_channels, nb_channels, nb_maps, gauss_std, norm=norm)
        self.decoder_recon = Decoder(ndims, nb_channels*4, in_channels, nb_channels, norm=norm)

    def forward(self, src, tgt, seg):
        src_embeddings = self.image_encoder(src)
        tgt_embeddings = self.image_encoder(tgt)
        
        im_recon = self.decoder_recon(src_embeddings[0], src_embeddings[1], src_embeddings[2])
       
        tgt_gauss_feat, tgt_gauss_mu, tgt_gauss_maps = self.pose_encoder(src_embeddings[2], tgt_embeddings[2], seg, src.shape[2:])

        tgt_gauss_map1 = torch.sum(tgt_gauss_maps[2], dim=1, keepdim=True)
        tgt_gauss_map2 = torch.sum(tgt_gauss_maps[1], dim=1, keepdim=True)
        tgt_gauss_map3 = torch.sum(tgt_gauss_maps[0], dim=1, keepdim=True)

        embeddings1 = src_embeddings[0] * (1-tgt_gauss_map1) + tgt_embeddings[0] * tgt_gauss_map1
        embeddings2 = src_embeddings[1] * (1-tgt_gauss_map2) + tgt_embeddings[1] * tgt_gauss_map2
        embeddings3 = src_embeddings[2] * (1-tgt_gauss_map3) + tgt_embeddings[2] * tgt_gauss_map3

        im_cross = self.decoder_recon(embeddings1, embeddings2, embeddings3)
        return {
            'recon': im_recon, 'cross': im_cross, 'gauss_mu': tgt_gauss_mu, 'gauss_maps': tgt_gauss_maps, 'gauss_feat': tgt_gauss_feat
        }
    
    def pose(self, src, tgt, seg):
        src_embeddings = self.image_encoder(src)
        tgt_embeddings = self.image_encoder(tgt)
        _, tgt_gauss_mu, tgt_gauss_maps = self.pose_encoder(src_embeddings[2], tgt_embeddings[2], seg, src.shape[2:])
        return tgt_gauss_mu, tgt_gauss_maps
        
        

class KeyPointer(nn.Module):
    def __init__(
        self,
        ndims=2,
        in_channels=1,
        nb_channels=16,
        nb_maps=1,
        gauss_std=0.1,
        norm='gn',
    ):
        super().__init__()

        self.encoder = Encoder(ndims, in_channels, nb_channels, norm=norm)
        self.conv_map = ConvBlock(ndims, nb_channels*4, 1, ksize=1, stride=1, padding=0, norm='none', act='none')

        self.gauss_std = gauss_std
        self.nb_maps = nb_maps

    def forward(self, x1, x2, mask, shape):
        mask = F.interpolate(mask, scale_factor=0.125, mode='nearest')
        x1_feat = x1
        x2_feat = x2
        x_feat = (x1_feat-x2_feat)**2
        x_feat = self.conv_map(x_feat*mask)
        
        attn = nms_kpts(x_feat, mask, d=5, num_points=self.nb_maps)

        if attn.shape[1]==0:
            gauss_mu = attn.permute(0,2,1)
            x_point = torch.relu(x_feat)*mask
            _, gauss_maps_tmp = self.get_keypoint_data_from_feature_map(x_point, [1/8., 1/4., 1/2.], self.gauss_std, shape=shape)
            gauss_maps = [torch.zeros_like(m).to(m.device) for m in gauss_maps_tmp]
        else:
            attn = self._get_gaussian_maps(attn.permute(0,2,1), [int(i/8) for i in shape], 1.0 / self.gauss_std)
            x_point = torch.relu(x_feat)*attn*mask
            gauss_mu, gauss_maps = self.get_keypoint_data_from_feature_map(x_point, [1/8., 1/4., 1/2.], self.gauss_std, shape=shape)
        return x_feat, gauss_mu, gauss_maps

    def get_keypoint_data_from_feature_map(self, point_map, map_sizes, gauss_std, shape):
        """Returns keypoint information from a feature map.
        Args:
            feature_map: [B, K, D, W, H] Tensor, should be activations from a convnet.
            gauss_std: float, the standard deviation of the gaussians to be put around
            the keypoints.
        Returns:
            a dict with keys:
            'centers': A tensor of shape [B, 3, K] of the center locations for each
                of the K keypoints.
            'heatmaps': A tensor of shape [B, K, D, W, H] of gaussian maps over the
                keypoints.
        """
        gauss_mu = self._get_keypoint_mus(point_map)
        gauss_maps = []
        dims = len(point_map.shape[2:])
        for map_size in map_sizes:
            map_size = [int(map_size*i) for i in shape]
            gauss_maps.append(self._get_gaussian_maps(gauss_mu, map_size, 1.0 / gauss_std))

        return gauss_mu, gauss_maps

    def _get_keypoint_mus(self, keypoint_features):
        """Returns the keypoint center points.
        Args:
            keypoint_features: A tensor of shape [B, K, F_d, F_w, F_h] where K is the number
            of keypoints to extract.
        Returns:
            A tensor of shape [B, 3, K] of the y, x center points of each keypoint. Each
            center point are in the range [-1, 1]^2. Note: the first element is the y
            coordinate, the second is the x coordinate.
        """
        dims = len(keypoint_features.shape[2:])
        if dims == 2:
            gauss_y = self._get_coord(keypoint_features, 2)
            gauss_x = self._get_coord(keypoint_features, 3)
            gauss_mu = torch.stack([gauss_y, gauss_x], dim=1)
        elif dims == 3:
            gauss_z = self._get_coord(keypoint_features, 2)
            gauss_y = self._get_coord(keypoint_features, 3)
            gauss_x = self._get_coord(keypoint_features, 4)
            gauss_mu = torch.stack([gauss_z, gauss_y, gauss_x], dim=1)
        return gauss_mu

    def _get_coord(self, features, axis):
        """Returns the keypoint coordinate encoding for the given axis.
        Args:
            features: A tensor of shape [B, K, F_d, F_w, F_h] where K is the number of
            keypoints to extract.
            axis: `int` which axis to extract the coordinate for. Has to be axis 1 or 2.
        Returns:
            A tensor of shape [B, K] containing the keypoint centers along the given
            axis. The location is given in the range [-1, 1].
        """
        if axis != 2 and axis != 3 and axis != 4:
            raise ValueError("Axis needs to be 2 or 3 or 4.")
        
        dims = len(features.shape[2:])
        if dims == 2:
            other_axis = [2, 3]
        elif dims == 3:
            other_axis = [2, 3, 4]
        other_axis.remove(axis)
        axis_size = features.shape[axis]

        # Compute the normalized weight for each row/column along the axis
        g_c_prob = torch.mean(features, dim=other_axis)

        # Linear combination of the interval [-1, 1] using the normalized weights to
        # give a single coordinate in the same interval [-1, 1]
        scale = torch.linspace(-1.0, 1.0, steps=axis_size, requires_grad=True, device=features.device).view([1, 1, axis_size])
        coordinate = torch.sum(g_c_prob * scale, dim=2)/(torch.sum(g_c_prob, dim=2)+1e-8)
        return coordinate

    def _get_gaussian_maps(self, mu, map_size, inv_std, power=2):
        """Transforms the keypoint center points to a gaussian masks."""
        dims = len(map_size)

        if dims == 2:
            mu_y, mu_x = mu[:,0,:].unsqueeze(-1).unsqueeze(-1), mu[:,1,:].unsqueeze(-1).unsqueeze(-1)

            y = torch.linspace(-1.0, 1.0, steps=map_size[0], requires_grad=True, device=mu_y.device).view([1, 1, map_size[0], 1])
            x = torch.linspace(-1.0, 1.0, steps=map_size[1], requires_grad=True, device=mu_x.device).view([1, 1, 1, map_size[1]])

            g_y = torch.pow(y - mu_y, power)
            g_x = torch.pow(x - mu_x, power)
            dist = (g_y + g_x) * np.power(inv_std, power)

        elif dims == 3:
            mu_z, mu_y, mu_x = mu[:,0,:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), mu[:,1,:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), mu[:,2,:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            z = torch.linspace(-1.0, 1.0, steps=map_size[0], requires_grad=True, device=mu_z.device).view([1, 1, map_size[0], 1, 1])
            y = torch.linspace(-1.0, 1.0, steps=map_size[1], requires_grad=True, device=mu_y.device).view([1, 1, 1, map_size[1], 1])
            x = torch.linspace(-1.0, 1.0, steps=map_size[2], requires_grad=True, device=mu_x.device).view([1, 1, 1, 1, map_size[2]])
            
            g_z = torch.pow(z - mu_z, power)
            g_y = torch.pow(y - mu_y, power)
            g_x = torch.pow(x - mu_x, power)
            dist = (g_z + g_y + g_x) * np.power(inv_std, power)
        g_zyx = torch.exp(-dist)

        return g_zyx


class Decoder(nn.Module):
    def __init__(
        self,
        ndims,
        in_channels=1,
        out_channels=1,
        nb_channels=16,
        norm='bn',
        outact='sigmoid',
    ):
        super().__init__()

        # Encoder
        nf = nb_channels
        self.render3 = nn.Sequential(
            ConvBlock(ndims, in_channels, nf*4, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            ConvBlock(ndims, nf*4, nf*4, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.render2 = nn.Sequential(
            ConvBlock(ndims, nf*6, nf*2, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            ConvBlock(ndims, nf*2, nf*2, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.render1 = nn.Sequential(
            ConvBlock(ndims, nf*3, nf, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            ConvBlock(ndims, nf, nf, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            nn.Upsample(scale_factor=2, mode='trilinear' if ndims==3 else 'bilinear', align_corners=True),
            ConvBlock(ndims, nf, out_channels, ksize=3, stride=1, padding=1, norm='none', act=outact),
        )

    def forward(self, x1, x2, x3):
        x = self.render3(x3)
        x = self.render2(torch.cat([x, x2], 1))
        x = self.render1(torch.cat([x, x1], 1))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        ndims,
        in_channels=1,
        nb_channels=32,
        norm='bn',
    ):
        super().__init__()

        # Encoder
        nf = nb_channels
        self.conv_b1 = nn.Sequential(
            ConvBlock(ndims, in_channels, nf, ksize=7, stride=2, padding=3, norm=norm, group=1, act='lrelu'),
            ConvBlock(ndims, nf, nf, ksize=3, stride=1, padding=1, norm=norm, group=nf//8, act='lrelu'),
        )
        nf *= 2
        self.conv_b2 = nn.Sequential(
            ConvBlock(ndims, nf//2, nf, ksize=3, stride=2, padding=1, norm=norm, group=nf//8, act='lrelu'),
            ConvBlock(ndims, nf, nf, ksize=3, stride=1, padding=1, norm=norm, group=nf//8, act='lrelu'),
        )
        nf *= 2
        self.conv_b3 = nn.Sequential(
            ConvBlock(ndims, nf//2, nf, ksize=3, stride=2, padding=1, norm=norm, group=nf//8, act='lrelu'),
            ConvBlock(ndims, nf, nf, ksize=3, stride=1, padding=1, norm=norm, group=nf//8, act='lrelu'),
        )
        nf *= 2
        self.conv_b4 = nn.Sequential(
            ConvBlock(ndims, nf//2, nf, ksize=3, stride=2, padding=1, norm=norm, group=nf//8, act='lrelu'),
            ConvBlock(ndims, nf, nf, ksize=3, stride=1, padding=1, norm=norm, group=nf//8, act='lrelu'),
        )

    def forward(self, x):
        x1 = self.conv_b1(x)
        x2 = self.conv_b2(x1)
        x3 = self.conv_b3(x2)
        x4 = self.conv_b4(x3)
        return [x1, x2, x3, x4]



def farthest_point_sampling(kpts, num_points):
    _, N, _ = kpts.size()
    ind = torch.zeros(num_points).long()
    ind[0] = torch.randint(N, (1,))
    dist = torch.sum((kpts - kpts[:, ind[0], :]) ** 2, dim=2)
    for i in range(1, num_points):
        ind[i] = torch.argmax(dist)
        dist = torch.min(dist, torch.sum((kpts - kpts[:, ind[i], :]) ** 2, dim=2))
       
    return kpts[:, ind, :], ind


def nms_kpts(img, mask, d=9, num_points=8):
    _, _, D, H, W = img.shape
    device = img.device
    dtype = img.dtype
    
    pad1 = d//2
    pad2 = d - pad1 - 1
    
    maxfeat = F.max_pool3d(F.pad(img, (pad2, pad1, pad2, pad1, pad2, pad1)), d, stride=1)
    structure_element = torch.tensor([[[0., 0,  0],
                                       [0,  1,  0],
                                       [0,  0,  0]],
                                      [[0,  1,  0],
                                       [1,  0,  1],
                                       [0,  1,  0]],
                                      [[0,  0,  0],
                                       [0,  1,  0],
                                       [0,  0,  0]]]).to(device)
    mask_eroded = (1 - F.conv3d(1 - mask.to(dtype), structure_element.unsqueeze(0).unsqueeze(0), padding=1).clamp_(0, 1)).bool()
    
    kpts = torch.nonzero(mask_eroded & (maxfeat == img)).unsqueeze(0).to(dtype)[:, :, 2:]

    if kpts.shape[1]>num_points:
        kpts = farthest_point_sampling(kpts, num_points)[0]
    else:
        num_points = kpts.shape[1]
    kpts = kpts.to(dtype=torch.int64)
    if num_points > 0:
        kpts = torch.cat([
            kpts[:,:,0:1]/img.shape[2]*2-1,
            kpts[:,:,1:2]/img.shape[3]*2-1,
            kpts[:,:,2:3]/img.shape[4]*2-1,
        ], dim=2)

    return kpts
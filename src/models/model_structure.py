import torch
import torch.nn as nn

import numpy as np

from models.layers import ConvBlock


class KN_S(nn.Module):
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
        self.decoder = Decoder(ndims, nb_channels*4, in_channels, nb_channels, norm=norm)

        self.conv_func1 = nn.Sequential(
            ConvBlock(ndims, nb_channels*4+nb_maps, nb_channels*4, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            ConvBlock(ndims, nb_channels*4, nb_channels*4, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
        )
        self.conv_func2 = nn.Sequential(
            ConvBlock(ndims, nb_channels*2+nb_maps, nb_channels*2, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            ConvBlock(ndims, nb_channels*2, nb_channels*2, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
        )
        self.conv_func3 = nn.Sequential(
            ConvBlock(ndims, nb_channels+nb_maps, nb_channels, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
            ConvBlock(ndims, nb_channels, nb_channels, ksize=3, stride=1, padding=1, norm=norm, act='lrelu'),
        )

    def forward(self, src, tgt):
        src_embeddings = self.image_encoder(src)
        tgt_embeddings = self.image_encoder(tgt)
        _, src_gauss_mu, src_gauss_maps = self.pose_encoder(src)
        _, tgt_gauss_mu, tgt_gauss_maps = self.pose_encoder(tgt)
        
        st_embeddings1 = self.conv_func3(torch.cat([src_embeddings[0], tgt_gauss_maps[3]], 1))
        st_embeddings2 = self.conv_func2(torch.cat([src_embeddings[1], tgt_gauss_maps[2]], 1))
        st_embeddings3 = self.conv_func1(torch.cat([src_embeddings[2], tgt_gauss_maps[1]], 1))
        im_src2tgt = self.decoder(st_embeddings1, st_embeddings2, st_embeddings3)

        ts_embeddings1 = self.conv_func3(torch.cat([tgt_embeddings[0], src_gauss_maps[3]], 1))
        ts_embeddings2 = self.conv_func2(torch.cat([tgt_embeddings[1], src_gauss_maps[2]], 1))
        ts_embeddings3 = self.conv_func1(torch.cat([tgt_embeddings[2], src_gauss_maps[1]], 1))
        im_tgt2src = self.decoder(ts_embeddings1, ts_embeddings2, ts_embeddings3)

        return {
            'src2tgt': im_src2tgt, 'tgt2src': im_tgt2src, 'gauss_mu_src': src_gauss_mu, 'gauss_maps_src': src_gauss_maps, 'gauss_mu_tgt': tgt_gauss_mu, 'gauss_maps_tgt': tgt_gauss_maps,
        }
        

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
        self.conv_map = ConvBlock(ndims, nb_channels*8, nb_maps, ksize=1, stride=1, padding=0, norm='none', act='none')

        self.gauss_std = gauss_std

    def forward(self, x):
        x_feat = self.encoder(x)[-1]
        x_point = self.conv_map(x_feat)
        
        gauss_mu, gauss_maps = self.get_keypoint_data_from_feature_map(x_point, [1/16., 1/8., 1/4., 1/2.], self.gauss_std, shape=x.shape[2:])
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
        g_c_prob = torch.softmax(g_c_prob, dim=2)

        # Linear combination of the interval [-1, 1] using the normalized weights to
        # give a single coordinate in the same interval [-1, 1]
        scale = torch.linspace(-1.0, 1.0, steps=axis_size, requires_grad=True, device=features.device).view([1, 1, axis_size])
        coordinate = torch.sum(g_c_prob * scale, dim=2)
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
            ConvBlock(ndims, nf, out_channels, ksize=3, stride=1, padding=1, norm='none', act='sigmoid'),
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
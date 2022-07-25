import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

#from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args


class ConditionalInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim=64):
        super().__init__()

        self.norm = nn.InstanceNorm3d(in_channel)

        self.style = nn.Linear(latent_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, latent_code):
        # style [batch_size, in_channels*2] => [batch_size, in_channels*2, 1, 1, 1]
        style = self.style(latent_code).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        gamma, beta = style.chunk(2, dim=1)

        out = self.norm(input)
        # out = input
        out = (1. + gamma) * out + beta

        return out


class PreActBlock_Conditional(nn.Module):
    """Pre-activation version of the BasicBlock + Conditional instance normalization"""
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False, latent_dim=64, mapping_fmaps=64):
        super(PreActBlock_Conditional, self).__init__()
        self.ai1 = ConditionalInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.ai2 = ConditionalInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        self.mapping = nn.Sequential(
            nn.Linear(1, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, latent_dim),
            nn.LeakyReLU(0.2)
        )

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x, reg_code):

        latent_fea = self.mapping(reg_code)

        out = F.leaky_relu(self.ai1(x, latent_fea), negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(self.ai2(out, latent_fea), negative_slope=0.2))

        out += shortcut
        return out


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        nf_in = 4
        nf = 32
        
        self.hr_c1 = ConvBlock(ndims, nf_in, nf, 2)
        self.hr_c2 = ConvBlock(ndims, nf, nf*2, 2)
        self.resblock_group_hr = self.resblock_seq(nf*2, bias_opt=False)
        self.hr_uc1 = ConvBlock(ndims, nf*3, nf)
        self.hr_uc2 = ConvBlock(ndims, nf+nf_in, nf)

        self.mr_c1 = ConvBlock(ndims, nf_in, nf*2, 2)
        self.mr_c2 = ConvBlock(ndims, nf*2, nf*4, 2)
        self.resblock_group_mr = self.resblock_seq(nf*4, bias_opt=False)
        self.mr_uc1 = ConvBlock(ndims, nf*6, nf*2)
        self.mr_uc2 = ConvBlock(ndims, nf*2+nf_in, nf)

        self.lr_c1 = ConvBlock(ndims, nf_in, nf*2, 2)
        self.lr_c2 = ConvBlock(ndims, nf*2, nf*4, 2)
        self.resblock_group_lr = self.resblock_seq(nf*4, bias_opt=False)
        self.lr_uc1 = ConvBlock(ndims, nf*6, nf*2)
        self.lr_uc2 = ConvBlock(ndims, nf*2+nf_in, nf)

        self.lr_merge = ConvBlock(ndims, nf*2, nf)
        self.mr_merge = ConvBlock(ndims, nf*2, nf)

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow_lr = Conv(nf, ndims, kernel_size=3, padding=1)
        # init flow layer with small weights and bias
        self.flow_lr.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_lr.weight.shape))
        self.flow_lr.bias = nn.Parameter(torch.zeros(self.flow_lr.bias.shape))

        self.flow_mr = Conv(nf, ndims, kernel_size=3, padding=1)
        # init flow layer with small weights and bias
        self.flow_mr.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_mr.weight.shape))
        self.flow_mr.bias = nn.Parameter(torch.zeros(self.flow_mr.bias.shape))

        self.flow_hr = Conv(nf, ndims, kernel_size=3, padding=1)
        # init flow layer with small weights and bias
        self.flow_hr.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_hr.weight.shape))
        self.flow_hr.bias = nn.Parameter(torch.zeros(self.flow_hr.bias.shape))

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_seg = layers.SpatialTransformer(inshape, mode='nearest')
        self.transformer2 = layers.SpatialTransformer([i//2 for i in inshape])
        self.transformer_seg2 = layers.SpatialTransformer([i//2 for i in inshape], mode='nearest')
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
    
    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.ModuleList([
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
            ]
        )
        return layer

    def net(self, reg_code, source_img, target_img):
        hr = torch.cat([source_img, target_img], dim=1)
        #mr = F.interpolate(hr, scale_factor=0.5, mode='trilinear', align_corners=True)
        #lr = F.interpolate(hr, scale_factor=0.25, mode='trilinear', align_corners=True)

        # high resolution
        e_hr1 = self.hr_c1(hr)
        x_hr = self.hr_c2(e_hr1)
        for i in range(len(self.resblock_group_hr)):
            if i % 2 == 0:
                x_hr = self.resblock_group_hr[i](x_hr, reg_code)
            else:
                x_hr = self.resblock_group_hr[i](x_hr)
        x_hr = self.upsample(x_hr)
        x_hr = self.hr_uc1(torch.cat([x_hr, e_hr1], dim=1))
        x_hr = self.upsample(x_hr)
        x_hr = self.hr_uc2(torch.cat([x_hr, hr], dim=1))
        x_hr = F.interpolate(x_hr, scale_factor=0.5, mode='trilinear', align_corners=True)

        #x_hr = self.mr_merge(torch.cat([x_hr, x_hr], dim=1))
        flow_hr = self.flow_hr(x_hr)
        return flow_hr, flow_hr, flow_hr

    def forward(self, reg_code, source_img, target_img, with_seg=False, sourceseg=None):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        flow_lr, flow_mr, flow_hr = self.net(reg_code, source_img, target_img)
        flow_lr_f = F.interpolate(flow_lr, scale_factor=2, mode='trilinear', align_corners=True)*2
        flow_mr_f = F.interpolate(flow_mr, scale_factor=2, mode='trilinear', align_corners=True)*2
        flow_hr_f = F.interpolate(flow_hr, scale_factor=2, mode='trilinear', align_corners=True)*2

        # warp image with flow field
        y_source_img_lr = self.transformer2(source_img, flow_lr)
        y_source_img_mr = self.transformer2(source_img, flow_mr)
        y_source_img_hr = self.transformer2(source_img, flow_hr)

        y_source_img_lr_f = self.transformer(source_img, flow_lr_f)
        y_source_img_mr_f = self.transformer(source_img, flow_mr_f)
        y_source_img_hr_f = self.transformer(source_img, flow_hr_f)

        if with_seg:
            y_source_seg_lr = self.transformer_seg2(sourceseg, flow_lr)
            y_source_seg_mr = self.transformer_seg2(sourceseg, flow_mr)
            y_source_seg_hr = self.transformer_seg2(sourceseg, flow_hr)
            return {
                'flow_lr': flow_lr, 'lr_img': y_source_img_lr, 'lr_seg': y_source_seg_lr, 'lr_img_f': y_source_img_lr_f,
                'flow_mr': flow_mr, 'mr_img': y_source_img_mr, 'mr_seg': y_source_seg_mr, 'mr_img_f': y_source_img_mr_f,
                'flow_hr': flow_hr, 'hr_img': y_source_img_hr, 'hr_seg': y_source_seg_hr, 'hr_img_f': y_source_img_hr_f,
            }
        else:
            return {
                'flow_lr': flow_lr, 'lr_img': y_source_img_lr, 'lr_img_f': y_source_img_lr_f,
                'flow_mr': flow_mr, 'mr_img': y_source_img_mr, 'mr_img_f': y_source_img_mr_f,
                'flow_hr': flow_hr, 'hr_img': y_source_img_hr, 'hr_img_f': y_source_img_hr_f,
            }


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main1 = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation1 = nn.LeakyReLU(0.2)
        self.main2 = Conv(out_channels, out_channels, 3, 1, 1)
        self.activation2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main1(x)
        out = self.activation1(out)
        out = self.main2(out)
        out = self.activation2(out)
        return out

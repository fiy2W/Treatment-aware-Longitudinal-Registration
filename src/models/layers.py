import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        ndims,
        in_channels,
        out_channels,
        ksize=3,
        stride=1,
        padding=1,
        dilation=1,
        group=1,
        norm='none',
        act='lrelu',
        bias=True,
    ):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=group, bias=bias)
        if act == 'none':
            self.activation = None
        elif act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif act == 'softmax':
            self.activation = nn.Softmax(1)

        # Init Normalization
        if norm == 'in':
            norm = getattr(nn, 'InstanceNorm%dd' % ndims)
            self.norm = norm(out_channels, affine=True, track_running_stats=True)
        elif norm == 'bn':
            norm = getattr(nn, 'BatchNorm%dd' % ndims)
            self.norm = norm(out_channels)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(num_groups=group, num_channels=out_channels)
        elif norm == 'none':
            self.norm = None

    def forward(self, x):
        out = self.main(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy import signal


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class GaussKernel(nn.Module):
    def __init__(
        self,
        gksize=5,
        gkstd=0.1,
        device=None,
    ):
        super().__init__()
        # 3d Gauss
        self.pad = (gksize - 1) // 2
        gkernel1d = signal.gaussian(gksize, std=gkstd)
        gkernel = np.outer(gkernel1d, np.outer(gkernel1d, gkernel1d)).reshape((gksize, gksize, gksize))
        gkernel = gkernel / np.sum(gkernel)
        self.weight = torch.FloatTensor(gkernel).view((1,1,gksize,gksize,gksize)).to(device=device)

    def forward(self, x):
        out = F.conv3d(x, self.weight, padding=self.pad)
        return out


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def	forward(self, input, target):
        N = target.size(0)
        smooth = 1e-5
        input_flat = input#torch.softmax(input, dim=1)
        target_flat = target

        intersection = input_flat * target_flat
        
        #w = 1 / (target_flat.sum((2,3,4))+smooth)
        #w = w / w.sum()
        loss = 2 * intersection.sum((2,3,4)) / (input_flat.sum((2,3,4)) + target_flat.sum((2,3,4)) + smooth)
        
        loss = torch.mean(1 - loss)

        return loss


class MutualInformation:
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes
      (e.g. probabilistic segmentaitons)
    More information/citation:
    - Courtney K Guo.
      Multi-modal image registration with unsupervised deep learning.
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, JE Iglesias, B Fischl, AV Dalca.
      Learning MRI Contrast-Agnostic Registration.
      ISBI: IEEE International Symposium on Biomedical Imaging, pp 899-903, 2021.
      https://doi.org/10.1109/ISBI48211.2021.9434113
    # TODO: add local MI by using patches. This is quite memory consuming, though.
    Includes functions that can compute mutual information between volumes,
      between segmentations, or between a volume and a segmentation map
    mi = MutualInformation()
    mi.volumes
    mi.segs
    mi.volume_seg
    mi.channelwise
    mi.maps
    """

    def __init__(self,
                 bin_centers=None,
                 nb_bins=None,
                 soft_bin_alpha=None,
                 min_clip=None,
                 max_clip=None):
        """
        Initialize the mutual information class
        Arguments below are related to soft quantizing of volumes, which is done automatically
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise())
        using these parameters
        Args:
            bin_centers (np.float32, optional): Array or list of bin centers. Defaults to None.
            nb_bins (int, optional):  Number of bins. Defaults to 16 if bin_centers
                is not specified.
            soft_bin_alpha (int, optional): Alpha in RBF of soft quantization. Defaults
                to `1 / 2 * square(sigma)`.
            min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
            max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
        """

        self.bin_centers = None
        if bin_centers is not None:
            self.bin_centers = torch.Tensor(bin_centers, dtype=torch.float32)
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins
        if bin_centers is None and nb_bins is None:
            self.nb_bins = 16

        self.min_clip = min_clip
        if self.min_clip is None:
            self.min_clip = -np.inf

        self.max_clip = max_clip
        if self.max_clip is None:
            self.max_clip = np.inf

        self.soft_bin_alpha = soft_bin_alpha
        if self.soft_bin_alpha is None:
            sigma_ratio = 0.5
            if self.bin_centers is None:
                sigma = sigma_ratio / (self.nb_bins - 1)
            else:
                sigma = sigma_ratio * torch.mean(bin_centers[1:]-bin_centers[:-1])#K.experimental.numpy.diff(bin_centers)
            self.soft_bin_alpha = 1 / (2 * np.square(sigma))
            #print(self.soft_bin_alpha)

    def volumes(self, x, y):
        """
        Mutual information for each item in a batch of volumes.
        Algorithm:
        - use neurite.utils.soft_quantize() to create a soft quantization (binning) of
          intensities in each channel
        - channelwise()
        Parameters:
            x and y:  [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        #tensor_channels_x = x.view(-1)#K.shape(x)[-1]
        #tensor_channels_y = y.view(-1)#K.shape(y)[-1]
        #msg = 'volume_mi requires two single-channel volumes. See channelwise().'
        #tf.debugging.assert_equal(tensor_channels_x, 1, msg)
        #tf.debugging.assert_equal(tensor_channels_y, 1, msg)

        # volume mi
        return torch.flatten(self.channelwise(x,y))#K.flatten(self.channelwise(x, y))

    def segs(self, x, y):
        """
        Mutual information between two probabilistic segmentation maps.
        Wraps maps()
        Parameters:
            x and y:  [bs, ..., nb_labels]
        Returns:
            Tensor of size [bs]
        """
        # volume mi
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """
        Mutual information between a volume and a probabilistic segmentation maps.
        Wraps maps()
        Parameters:
            x and y: a volume and a probabilistic (soft) segmentation. Either:
              - x: [bs, ..., 1] and y: [bs, ..., nb_labels], Or:
              - x: [bs, ..., nb_labels] and y: [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = x.view(-1)#K.shape(x)[-1]
        tensor_channels_y = y.view(-1)#K.shape(y)[-1]
        msg = 'volume_seg_mi requires one single-channel volume.'
        assert torch.minimum((tensor_channels_x,tensor_channels_y))==1, msg#tf.debugging.assert_equal(tf.minimum(tensor_channels_x, tensor_channels_y), 1, msg)
        # otherwise we don't know which one is which
        msg = 'volume_seg_mi requires one multi-channel segmentation.'
        assert torch.maximum(tensor_channels_x,tensor_channels_y)==1, msg
        #tf.debugging.assert_greater(tf.maximum(tensor_channels_x, tensor_channels_y), 1, msg)

        # transform volume to soft-quantized volume
        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[..., 0])                       # [bs, ..., B]
        else:
            y = self._soft_sim_map(y[..., 0])                       # [bs, ..., B]

        return self.maps(x, y)  # [bs]

    def channelwise(self, x, y):
        """
        Mutual information for each channel in x and y. Thus for each item and channel this
        returns retuns MI(x[...,i], x[...,i]). To do this, we use neurite.utils.soft_quantize() to
        create a soft quantization (binning) of the intensities in each channel
        Parameters:
            x and y:  [bs, ..., C]
        Returns:
            Tensor of size [bs, C]
        """
        # check shapes
        x = x.permute(0,2,3,4,1)
        y = y.permute(0,2,3,4,1)
        tensor_shape_x = x.size()#K.shape(x)
        tensor_shape_y = y.size()#K.shape(y)
        assert tensor_shape_x==tensor_shape_y, 'volume shapes do not match'
        #tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y, 'volume shapes do not match')

        # reshape to [bs, V, C]
        if tensor_shape_x[0] != 3:
            x = torch.reshape(x, [tensor_shape_x[0], -1, tensor_shape_x[-1]])                             # [bs, V, C]
            y = torch.reshape(y, [tensor_shape_x[0], -1, tensor_shape_x[-1]])                             # [bs, V, C]

        # move channels to first dimension
        ndims_k = len(x.shape)
        permute = [ndims_k - 1] + list(range(ndims_k - 1))
        cx = x.permute(permute)#K.transpose(x, permute)                                # [C, bs, V]
        cy = y.permute(permute)#K.transpose(y, permute)                                # [C, bs, V]

        # soft quantize
        cxq = self._soft_sim_map(cx)                                  # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)                                  # [C, bs, V, B]

        # get mi
        map_fn = lambda x: self.maps(*x)
        cout = []
        for i in range(cxq.shape[0]):  # C
            cout.append(self.maps(cxq[i, :, :, :], cyq[i, :, :, :]))
        cout = torch.stack(cout, 0)
        # cout = tf.map_fn(map_fn, [cxq, cyq], dtype=tf.float32)       # [C, bs]

        # permute back
        return cout.permute(1,0)#K.transpose(cout, [1, 0])                            # [bs, C]

    def maps(self, x, y):
        """
        Computes mutual information for each entry in batch, assuming each item contains
        probability or similarity maps *at each voxel*. These could be e.g. from a softmax output
        (e.g. when performing segmentaiton) or from soft_quantization of intensity image.
        Note: the MI is computed separate for each item in the batch, so the joint probabilities
        might be  different across inputs. In some cases, computing MI across the whole batch
        might be desireable (TODO).
        Parameters:
            x and y are probability maps of size [bs, ..., B], where B is the size of the
              discrete probability domain grid (e.g. bins/labels). B can be different for x and y.
        Returns:
            Tensor of size [bs]
        """

        # check shapes
        tensor_shape_x = x.size()#K.shape(x)
        tensor_shape_y = y.size()#K.shape(y)
        #tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y)
        #tf.debugging.assert_non_negative(x)
        # tf.debugging.assert_non_negative(y)

        eps = 1e-6#K.epsilon()

        # reshape to [bs, V, B]
        if tensor_shape_x[0] != 3:
            # new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = torch.reshape(x, (tensor_shape_x[0], -1, tensor_shape_x[-1]))                             # [bs, V, B1]
            y = torch.reshape(y, (tensor_shape_x[0], -1, tensor_shape_x[-1]))#tf.reshape(y, new_shape)                             # [bs, V, B2]

        # joint probability for each batch entry
        x_trans = x.permute(0,2,1)#tf.transpose(x, (0, 2, 1))                         # [bs, B1, V]
        pxy = torch.matmul(x_trans,y)#K.batch_dot(x_trans, y)                                # [bs, B1, B2]
        pxy = pxy / (torch.sum(pxy, dim=[1, 2], keepdim=True) + eps) #pxy / (K.sum(pxy, axis=[1, 2], keepdims=True) + eps)   # [bs, B1, B2]

        # x probability for each batch entry
        px = torch.sum(x, 1, keepdim=True)                              # [bs, 1, B1]
        px = px / (torch.sum(px, 2, keepdim=True) + eps)                # [bs, 1, B1]

        # y probability for each batch entry
        py = torch.sum(y, 1, keepdim=True)                              # [bs, 1, B2]
        py = py / (torch.sum(py, 2, keepdim=True) + eps)                # [bs, 1, B2]

        # independent xy probability
        px_trans = px.permute(0,2,1)#K.permute_dimensions(px, (0, 2, 1))               # [bs, B1, 1]
        pxpy = torch.matmul(px_trans,py)#K.batch_dot(px_trans, py)                             # [bs, B1, B2]
        pxpy_eps = pxpy + eps

        # mutual information
        log_term = torch.log(pxy / pxpy_eps + eps)                       # [bs, B1, B2]
        mi = torch.sum(pxy * log_term, dim=[1, 2])                      # [bs]
        return mi

    def _soft_log_sim_map(self, x):
        """
        soft quantization of intensities (values) in a given volume
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image.
        Returns:
            volume with one more dimension [bs, ..., B]
        """

        return soft_quantize(x,
                             alpha=self.soft_bin_alpha,
                             bin_centers=self.bin_centers,
                             nb_bins=self.nb_bins,
                             min_clip=self.min_clip,
                             max_clip=self.max_clip,
                             return_log=True)               # [bs, ..., B]

    def _soft_sim_map(self, x):
        """
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image.
        Returns:
            volume with one more dimension [bs, ..., B]
        """
        return soft_quantize(x,
                             alpha=self.soft_bin_alpha,
                             bin_centers=self.bin_centers,
                             nb_bins=self.nb_bins,
                             min_clip=self.min_clip,
                             max_clip=self.max_clip,
                             return_log=False)              # [bs, ..., B]

    def _soft_prob_map(self, x):#_soft_prob_map(self, x, **kwargs)
        """
        normalize a soft_quantized volume at each voxel, so that each voxel now holds a prob. map
        Parameters:
            x [bs, ..., B]: soft quantized volume
        Returns:
            x [bs, ..., B]: renormalized so that each voxel adds to 1 across last dimension
        """
        eps = 1e-6
        x_hist = self._soft_sim_map(x)#(x, **kwargs)                      # [bs, ..., B]
        x_hist_sum = torch.sum(x_hist, -1, keepdim=True)   # [bs, ..., B]
        x_prob = x_hist / (x_hist_sum + eps)                               # [bs, ..., B]
        return x_prob

    def loss(self, y_true, y_pred):
        return -self.volumes(y_true, y_pred)


def soft_quantize(x,
                  bin_centers=None,
                  nb_bins=16,
                  alpha=1,
                  min_clip=-np.inf,
                  max_clip=np.inf,
                  return_log=False):
    """
    (Softly) quantize intensities (values) in a given volume, based on RBFs.
    In numpy this (hard quantization) is called "digitize".
    Specify bin_centers OR number of bins
        (which will estimate bin centers based on a heuristic using the min/max of the image)
    Algorithm:
    - create (or obtain) a set of bins
    - for each array element, that value v gets assigned to all bins with
        a weight of exp(-alpha * (v - c)), where c is the bin center
    - return volume x nb_bins
    Parameters:
        x [bs, ...]: intensity image.
        bin_centers (np.float32 or list, optional): bin centers for soft histogram.
            Defaults to None.
        nb_bins (int, optional): number of bins, if bin_centers is not specified.
            Defaults to 16.
        alpha (int, optional): alpha in RBF.
            Defaults to 1.
        min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
        max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
        return_log (bool, optional): [description]. Defaults to False.
    Returns:
        tf.float32: volume with one more dimension [bs, ..., B]
    If you find this function useful, please cite the original paper this was written for:
        M Hoffmann, B Billot, JE Iglesias, B Fischl, AV Dalca.
        Learning MRI Contrast-Agnostic Registration.
        ISBI: IEEE International Symposium on Biomedical Imaging, pp 899-903, 2021.
        https://doi.org/10.1109/ISBI48211.2021.9434113
    """

    if bin_centers is not None:
        bin_centers = torch.from_numpy(bin_centers, torch.float32)
        assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
        nb_bins = bin_centers.shape[0]
    else:
        if nb_bins is None:
            nb_bins = 16
        # get bin centers dynamically
        # TODO: perhaps consider an option to quantize by percentiles:
        #   minval = tfp.stats.percentile(x, 1)
        #   maxval = tfp.stats.percentile(x, 99)
        minval = torch.min(x)#K.min(x)
        maxval = torch.max(x)#K.max(x)
        bin_centers = torch.linspace(minval.item(),maxval.item(),nb_bins)#tf.linspace(minval, maxval, nb_bins)

    # clipping at bin values
    x = torch.unsqueeze(x,dim=-1)#x[..., tf.newaxis]  # [..., 1]
    x = torch.clamp(x,min_clip,max_clip)#tf.clip_by_value(x, min_clip, max_clip)

    # reshape bin centers to be (1, 1, .., B)
    new_shape = [1] * (len(x.shape) - 1) + [nb_bins]
    bin_centers = torch.reshape(bin_centers,[1] * (len(x.shape) - 1) + [nb_bins])#K.reshape(bin_centers, new_shape)  # [1, 1, ..., B]

    # compute image terms
    # TODO: need to go to log space? not sure
    bin_diff = torch.square(x-bin_centers.to(device=x.device))#K.square(x - bin_centers)  # [..., B]
    log = -alpha * bin_diff  # [..., B]

    if return_log:
        return log  # [..., B]
    else:
        return torch.exp(log)#K.exp(log)  # [..., B]
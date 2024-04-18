import numpy as np

import torch
from torch import nn

from torch.nn import SyncBatchNorm, SiLU


def normalize_weight(log_weight_norm: nn.Parameter, weight: nn.Parameter) -> nn.Parameter:
    """ applies weight normalization """

    if log_weight_norm is not None:

        n = torch.exp(log_weight_norm)
        wn = torch.sqrt(torch.sum(torch.pow(weight, 2), dim=[1, 2, 3], keepdim=True))  # norm(w)
        return n * weight / (wn + 1e-5)
    else:
        return weight


class Conv2D(nn.Conv2d):
    """
    Custom implementation with weight normalization.
    TODO replace with native torch
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False, weight_norm: bool = True):
        """
        removed param "data_init" from original implementation, since was always set to False
        """

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.log_weight_norm = None

        if weight_norm:
            # initialize log weight norm parameter
            self.log_weight_norm = nn.Parameter(
                torch.log(1e-2 +
                          torch.sqrt(
                              torch.sum(
                                  torch.pow(self.weight, 2),
                                  dim=[1, 2, 3], keepdim=True)
                          )
                          ),
                requires_grad=True)

        self.weight_normalized = normalize_weight(self.log_weight_norm, self.weight)

    def forward(self, x: torch.Tensor):
        self.weight_normalized = normalize_weight(self.log_weight_norm, self.weight)
        return nn.functional.conv2d(x, self.weight_normalized, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)


class ARConv2d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False,
                 zero_diag: bool = False, mirror: bool = False):
        """
        Conv 2D with applied mask on each kernel
        """

        def channel_mask():

            assert in_channels % out_channels == 0 or out_channels % in_channels == 0
            assert groups == 1 or groups == in_channels

            mask = None

            if groups == 1:
                mask = torch.ones([out_channels, in_channels], dtype=torch.float32)
                if out_channels >= in_channels:
                    ratio = out_channels // in_channels
                    for i in range(in_channels):
                        mask[i * ratio:(i + 1) * ratio, i + 1:] = 0
                        if zero_diag:
                            mask[i * ratio:(i + 1) * ratio, i:i + 1] = 0
                else:
                    ratio = in_channels // out_channels
                    for i in range(out_channels):
                        mask[i:i + 1, (i + 1) * ratio:] = 0
                        if zero_diag:
                            mask[i:i + 1, i * ratio:(i + 1) * ratio:] = 0

            elif groups == in_channels:
                mask = torch.ones([out_channels, in_channels // groups], dtype=torch.float32)
                if zero_diag:
                    mask = 0. * mask

            return mask

        def create_conv_mask():
            """
            create the boolean mask for kernel
            """

            m = (kernel_size - 1) // 2
            mask = torch.ones([out_channels, in_channels // groups, kernel_size, kernel_size], dtype=torch.float32)
            mask[:, :, m:, :] = 0
            mask[:, :, m, :m] = 1
            mask[:, :, m, m] = channel_mask()
            if mirror:
                mask = torch.flip(mask, dims=[2, 3])  # TODO original made a copy, why ?
            return mask

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        assert kernel_size % 2 == 1, 'kernel size should be an odd value.'
        self.mask = create_conv_mask()

        # init weight normalization parameters
        self.log_weight_norm = nn.Parameter(
            torch.log(1e-2 +
                      torch.sqrt(
                          torch.sum(
                              torch.pow(self.weight * self.mask, 2),
                              dim=[1, 2, 3], keepdim=True
                          )
                      )
                      ),
            requires_grad=True)

        self.weight_normalized = normalize_weight(self.log_weight_norm, self.weight)

    def forward(self, x):

        self.weight_normalized = normalize_weight(self.log_weight_norm, self.weight)
        return nn.functional.conv2d(x, self.weight_normalized, self.bias, self.stride, self.padding,
                                    self.dilation, self.groups)


class SE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Squeeze and Excitation module, at the end of Residual Cells.
        :param in_channels:
        :param out_channels:
        """

        super().__init__()

        hidden_channels = max(out_channels // 16, 4)

        self.linear_1 = nn.Linear(in_channels, hidden_channels)
        self.linear_2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape

        # gate
        se = torch.mean(x, dim=[2, 3])
        se = nn.functional.relu(self.linear_1(se))
        se = nn.functional.sigmoid(self.linear_2(se))
        se = se.view(b, c, 1, 1)

        return x * se


class SkipDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        residual skip connection in case of downscaling.
        :param in_channels:
        :param out_channels: must be divisible by 4
        :param stride: if 2, downsampling is performed.
        """

        assert out_channels % 4 == 0, f"out_channels must be divisible by 4, but you passed: {out_channels}"

        super().__init__()

        # each conv reduces to 1/4 of the channels
        # TODO try to replace this with a normal convolution and see what happens...
        # self.conv_1 = Conv2D(in_channels, out_channels // 4, kernel_size=1, stride=stride, bias=True, weight_norm=True)
        # self.conv_2 = Conv2D(in_channels, out_channels // 4, kernel_size=1, stride=stride, bias=True, weight_norm=True)
        # self.conv_3 = Conv2D(in_channels, out_channels // 4, kernel_size=1, stride=stride, bias=True, weight_norm=True)
        # self.conv_4 = Conv2D(in_channels, out_channels // 4, kernel_size=1, stride=stride, bias=True, weight_norm=True)
        self.conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, bias=True, weight_norm=True)

    def forward(self, x):
        out = torch.nn.functional.silu(x)

        # conv1 = self.conv_1(out)
        # conv2 = self.conv_2(out[:, :, 1:, 1:])
        # conv3 = self.conv_3(out[:, :, :, 1:])
        # conv4 = self.conv_4(out[:, :, 1:, :])
        #
        # out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        out = self.conv(out)

        return out


class SkipUp(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, weight_norm=True)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class ResidualCellEncoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downsampling: bool, use_SE: bool):
        """
        FIG 4.B of the paper
        :param in_channels:
        :param out_channels:
        :param downsampling:
        :param use_SE:
        """

        super().__init__()

        # skip connection has convs if downscaling
        if downsampling:
            stride = 2
            self.skip_connection = SkipDown(in_channels, out_channels, stride)
        else:
            stride = 1
            self.skip_connection = nn.Identity()

        # (BN - SWISH) + conv 3x3 + (BN - SWISH) + conv 3x3 + SE
        # downsampling in the first conv, depending on stride
        self.residual = nn.Sequential(
            SyncBatchNorm(in_channels, eps=1e-5, momentum=0.05),
            SiLU(),
            Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, weight_norm=True),
            SyncBatchNorm(out_channels, eps=1e-5, momentum=0.05),
            SiLU(),
            Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, weight_norm=True)
        )

        if use_SE:
            self.residual.append(SE(out_channels, out_channels))

    def forward(self, x: torch.Tensor):

        residual = 0.1 * self.residual(x)
        x = self.skip_connection(x)

        return x + residual


class ResidualCellDecoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, upsampling: bool, use_SE: bool, hidden_mul: int = 6):
        """
        Figure 3a of the paper.

        :param in_channels:
        :param out_channels:
        :param upsampling:
        :param use_SE:
        """

        super().__init__()

        if upsampling:
            self.skip_connection = SkipUp(in_channels, out_channels, stride=1)
        else:
            self.skip_connection = nn.Identity()

        self.use_se = use_SE

        hidden_dim = in_channels * hidden_mul

        residual = [nn.UpsamplingNearest2d(scale_factor=2)] if upsampling else []

        residual += [
            SyncBatchNorm(in_channels, eps=1e-5, momentum=0.05),
            Conv2D(in_channels, hidden_dim, kernel_size=1, bias=False, weight_norm=False),
            SyncBatchNorm(hidden_dim, eps=1e-5, momentum=0.05),
            SiLU(),
            Conv2D(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False, weight_norm=False),
            SyncBatchNorm(hidden_dim, eps=1e-5, momentum=0.05),
            SiLU(),
            Conv2D(hidden_dim, out_channels, kernel_size=1, bias=False, weight_norm=False),
            SyncBatchNorm(out_channels, eps=1e-5, momentum=0.05)
        ]

        if self.use_se:
            residual.append(SE(out_channels, out_channels))

        self.residual = nn.Sequential(*residual)

    def forward(self, x: torch.Tensor):

        residual = 0.1 * self.residual(x)
        x = self.skip_connection(x)

        return x + residual


class EncCombinerCell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = Conv2D(in_channels, out_channels, kernel_size=1, bias=True, weight_norm=True)

    def forward(self, x_enc, x_dec):
        """
        combine encoder ad decoder features
        """

        x_dec = self.conv(x_dec)
        out = x_enc + x_dec
        return out


class DecCombinerCell(nn.Module):
    def __init__(self, feature_channels: int, z_channels: int, out_channels: int):
        """
        combine feature + noise channels during decoding / generation
        """

        super().__init__()

        self.conv = Conv2D(feature_channels + z_channels, out_channels, kernel_size=1, bias=True, weight_norm=True)

    def forward(self, x, z):
        out = torch.cat([x, z], dim=1)
        out = self.conv(out)
        return out


class ELUConv(nn.Module):
    """ReLU + Conv2d + BN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, dilation: int = 1,
                 zero_diag: bool = True, weight_init_coeff: int = 1.0, mirror: bool = False):
        super().__init__()
        self.conv_0 = ARConv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=True,
                               dilation=dilation, zero_diag=zero_diag, mirror=mirror)

        # change the initialized log weight norm
        self.conv_0.log_weight_norm.data += np.log(weight_init_coeff)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = nn.functional.elu(x)
        out = self.conv_0(out)
        return out


class ARInvertedResidual(nn.Module):
    def __init__(self, in_z: int, dilation: int = 1, kernel_size: int = 5, mirror=False):
        super().__init__()

        self.hidden_dim = int(round(in_z * 6))
        padding = dilation * (kernel_size - 1) // 2
        layers = []
        layers.extend([ARConv2d(in_z, self.hidden_dim, kernel_size=3, padding=1, mirror=mirror, zero_diag=True),
                       nn.ELU(inplace=True)])
        layers.extend([ARConv2d(self.hidden_dim, self.hidden_dim, groups=self.hidden_dim, kernel_size=kernel_size,
                                padding=padding, dilation=dilation, mirror=mirror, zero_diag=False),
                       nn.ELU(inplace=True)])
        self.conv_z = nn.Sequential(*layers)

    def forward(self, z):
        return self.conv_z(z)


class NFCell(nn.Module):
    def __init__(self, num_z: int, mirror: bool):
        super().__init__()

        # couple of convolution with mask applied on kernel
        self.conv = ARInvertedResidual(num_z, mirror=mirror)

        # 0.1 helps bring mu closer to 0 initially
        self.mu = ELUConv(self.conv.hidden_dim, num_z, kernel_size=1, padding=0, zero_diag=False,
                          weight_init_coeff=0.1, mirror=mirror)

    def forward(self, z):
        s = self.conv(z)

        mu = self.mu(s)
        new_z = (z - mu)

        return new_z


class NFBlock(nn.Module):
    def __init__(self, num_z: int):
        super().__init__()

        self.cell1 = NFCell(num_z, mirror=False)
        self.cell2 = NFCell(num_z, mirror=True)

    def forward(self, z):
        new_z = self.cell1(z)
        new_z = self.cell2(new_z)

        return new_z

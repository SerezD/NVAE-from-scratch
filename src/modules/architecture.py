from einops import rearrange

import torch
from torch import nn
from torch.nn import SyncBatchNorm, SiLU, Conv2d
from torch.nn.utils.parametrizations import weight_norm


class MaskedConv2d(nn.Conv2d):

    def __init__(self, mirror: bool, zero_diag: bool, *args, **kwargs):

        # init standard conv
        super().__init__(*args, **kwargs)

        # mask has same shape as kernel
        mask = torch.ones_like(self.weight)
        _, _, h, w = mask.shape

        # fill zeros where needed (moving on single dim and converting back at the end)
        mask = rearrange(mask, 'a b h w -> a b (h w)')
        half = ((h * w) // 2) + int(zero_diag)
        mask[:, :, half:] = 0
        if mirror:
            mask = torch.flip(mask, dims=(2,))
        mask = rearrange(mask, 'a b (h w) -> a b h w', h=h, w=w)

        self.register_buffer('mask', mask)

    def forward(self, x):
        with torch.no_grad():
            self.weight = nn.Parameter(self.weight * self.mask)

        return super().forward(x)


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

        super().__init__()

        self.conv = weight_norm(Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):

        out = torch.nn.functional.silu(x)
        out = self.conv(out)

        return out


class SkipUp(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = weight_norm(Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))

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
            weight_norm(Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)),
            SyncBatchNorm(out_channels, eps=1e-5, momentum=0.05),
            SiLU(),
            weight_norm(Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
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
            Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            SyncBatchNorm(hidden_dim, eps=1e-5, momentum=0.05),
            SiLU(),
            Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False),
            SyncBatchNorm(hidden_dim, eps=1e-5, momentum=0.05),
            SiLU(),
            Conv2d(hidden_dim,  out_channels, kernel_size=1, bias=False),
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

        self.conv = weight_norm(Conv2d(in_channels, out_channels, kernel_size=1, bias=True))

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

        self.conv = weight_norm(Conv2d(feature_channels + z_channels, out_channels, kernel_size=1, bias=True))

    def forward(self, x, z):
        out = torch.cat([x, z], dim=1)
        out = self.conv(out)
        return out


class NFCell(nn.Module):
    def __init__(self, num_z: int, mirror: bool):
        super().__init__()

        hidden_dim = int(num_z * 6)
        layers = [
            MaskedConv2d(mirror=mirror, zero_diag=True, in_channels=num_z, out_channels=hidden_dim,
                         kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            MaskedConv2d(mirror=mirror, zero_diag=False, in_channels=hidden_dim, out_channels=hidden_dim,
                         kernel_size=5, padding=2, groups=hidden_dim),
            nn.ELU(inplace=True),
            MaskedConv2d(zero_diag=False, mirror=mirror, in_channels=hidden_dim, out_channels=num_z,
                         kernel_size=1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return z - self.layers(z)


class NFBlock(nn.Module):
    def __init__(self, num_z: int):
        super().__init__()

        self.cell1 = NFCell(num_z, mirror=False)
        self.cell2 = NFCell(num_z, mirror=True)

    def forward(self, z):
        new_z = self.cell1(z)
        new_z = self.cell2(new_z)

        return new_z

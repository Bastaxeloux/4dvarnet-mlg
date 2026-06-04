import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels, max_groups):
    """Return a valid GroupNorm group count for the requested channel count."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, norm_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(_norm_groups(dim_out, norm_groups), dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, norm_groups=8, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = ConvNormAct(dim_in, dim_out, kernel_size=kernel_size, norm_groups=norm_groups)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_norm_groups(dim_out, norm_groups), dim_out)
        self.shortcut = (
            nn.Identity()
            if dim_in == dim_out
            else nn.Conv2d(dim_in, dim_out, kernel_size=1)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out, norm_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(_norm_groups(dim_out, norm_groups), dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def _make_blocks(dim_in, dim_out, blocks_per_level, kernel_size, norm_groups, dropout):
    blocks = [
        ResidualBlock(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            norm_groups=norm_groups,
            dropout=dropout,
        )
    ]
    for _ in range(1, blocks_per_level):
        blocks.append(
            ResidualBlock(
                dim_out,
                dim_out,
                kernel_size=kernel_size,
                norm_groups=norm_groups,
                dropout=dropout,
            )
        )
    return nn.Sequential(*blocks)


class ResUNetPriorCost(nn.Module):
    """
    Residual U-Net prior for SST 4D-VarNet solvers.

    This module keeps the same prior interface as BilinReconstructorPriorCost:
    forward_reconstructor(x_obs) reconstructs T SST channels, and forward()
    measures ||state - Phi([state, covariates])||^2.
    """
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        depth=3,
        blocks_per_level=1,
        kernel_size=3,
        norm_groups=8,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("ResUNetPriorCost requires depth >= 2")
        if blocks_per_level < 1:
            raise ValueError("ResUNetPriorCost requires blocks_per_level >= 1")

        self.dim_out = dim_out
        self.depth = depth

        channels = [dim_hidden * (2 ** i) for i in range(depth)]
        self.stem = ConvNormAct(dim_in, channels[0], kernel_size=kernel_size, norm_groups=norm_groups)

        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        for level, dim in enumerate(channels):
            self.encoder.append(
                _make_blocks(
                    dim,
                    dim,
                    blocks_per_level,
                    kernel_size,
                    norm_groups,
                    dropout,
                )
            )
            if level < depth - 1:
                self.downs.append(Downsample(dim, channels[level + 1], norm_groups=norm_groups))

        self.decoder = nn.ModuleList()
        for level in range(depth - 2, -1, -1):
            dim_in_level = channels[level + 1] + channels[level]
            self.decoder.append(
                _make_blocks(
                    dim_in_level,
                    channels[level],
                    blocks_per_level,
                    kernel_size,
                    norm_groups,
                    dropout,
                )
            )

        self.head = nn.Conv2d(channels[0], dim_out, kernel_size=1)

    def forward_reconstructor(self, x_obs):
        x_obs = torch.nan_to_num(x_obs, nan=0.0)

        x = self.stem(x_obs)
        skips = []
        for level, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            skips.append(x)
            if level < len(self.downs):
                x = self.downs[level](x)

        for decoder_block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)

        return self.head(x)

    def forward(self, state, batch):
        T = self.dim_out
        covariables_and_spatial = batch.input[:, T:, :, :]
        dynamic_input = torch.cat([state, covariables_and_spatial], dim=1)
        dynamic_input = torch.nan_to_num(dynamic_input, nan=0.0)
        reconstructed = self.forward_reconstructor(dynamic_input)
        return F.mse_loss(state, reconstructed)

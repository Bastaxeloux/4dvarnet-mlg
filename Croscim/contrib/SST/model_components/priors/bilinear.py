import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinReconstructorPriorCost(nn.Module):
    """
    Bilinear Reconstructor: Takes input channels and reconstructs T channels (SST).

    Layout d'entrée actuel (8*T + 4 canaux, support du prior dynamique Phi(state)):
        [fusion_masquee (T) | slstr_std (T) | aasti_std (T)
         | avhrr_av (T) | avhrr_std (T) | pmw_av (T) | pmw_std (T)
         | sea_ice_fraction (T) | spatial (4)]

        - dim_in : 124 (x10), 76 (x3), 44 (x1)
        - dim_out: 15  (x10), 9  (x3), 5  (x1)

    forward() receives [state, covariates] rather than fixed batch.input,
    allowing a dynamic prior Phi(state) that evolves during GradSolver steps.
    """
    def __init__(self, dim_in, dim_hidden, dim_out, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):
        super().__init__()
        self.nt = nt
        self.bilin_quad = bilin_quad
        self.dim_out = dim_out

        self.conv_in = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.bilin_1 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_21 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward_reconstructor(self, x_obs):
        """
        Reconstruct SST (T channels) from observations (dim_in channels).

        x_obs: Input observations (B, dim_in, H, W)
               Structure: [fusion_masquee (0:T), satellites, covariates, spatial]
        returns: Reconstructed SST (B, T, H, W)
        """
        x_obs = x_obs.nan_to_num(nan=0.0)

        x = self.down(x_obs)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state, batch):
        """
        Prior cost: dynamic Phi(state), measuring ||state - Phi([state, covariates])||^2.
        """
        T = self.dim_out

        covariables_and_spatial = batch.input[:, T:, :, :]
        dynamic_input = torch.cat([state, covariables_and_spatial], dim=1)
        dynamic_input = torch.nan_to_num(dynamic_input, nan=0.0)

        reconstructed = self.forward_reconstructor(dynamic_input)

        return F.mse_loss(state, reconstructed)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTPriorCost(nn.Module):
    """
    Vision Transformer prior for SST 4D-VarNet solvers.

    Same prior interface as BilinReconstructorPriorCost/ResUNetPriorCost:
    forward_reconstructor(x_obs) reconstructs T SST channels via patch
    embedding + transformer encoder + linear unpatchify head, and forward()
    measures ||state - Phi([state, covariates])||^2.

    Patchify is done with a strided conv (kernel=stride=patch_size), so
    self-attention runs over (img_size/patch_size)^2 tokens rather than over
    full-resolution feature maps. This keeps activation memory far below a
    full-resolution conv U-Net, which matters here: the solver unrolls the
    prior n_step times with create_graph=True for the variational gradient.
    """
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        patch_size=16,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        img_size=256,
        **kwargs,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if dim_hidden % num_heads != 0:
            raise ValueError("dim_hidden must be divisible by num_heads")

        self.dim_out = dim_out
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        num_patches = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim_hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=num_heads,
            dim_feedforward=int(dim_hidden * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim_hidden)
        self.head = nn.Linear(dim_hidden, dim_out * patch_size * patch_size)

    def forward_reconstructor(self, x_obs):
        """
        Reconstruct SST (T channels) from observations (dim_in channels).

        x_obs: Input observations (B, dim_in, H, W)
        returns: Reconstructed SST (B, T, H, W)
        """
        x_obs = torch.nan_to_num(x_obs, nan=0.0)
        B = x_obs.shape[0]
        p, g = self.patch_size, self.grid_size

        x = self.patch_embed(x_obs)            # (B, dim_hidden, g, g)
        x = x.flatten(2).transpose(1, 2)        # (B, g*g, dim_hidden)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = self.head(x)                        # (B, g*g, dim_out*p*p)

        x = x.view(B, g, g, self.dim_out, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, self.dim_out, g * p, g * p)
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

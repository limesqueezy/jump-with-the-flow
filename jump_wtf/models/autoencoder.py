import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torchcfm.models.unet import UNetModel
import torch.nn.functional as F

# UNetModel is the wrapper from torchcfm
class UNetModelWrapper_encoder(UNetModel):
    """
    UNet encoder with optional bottleneck augmentation.
      • bottleneck=False → [t, flat(x), flat(UNet_out)]
      • bottleneck=True  → [t, flat(x), flat(UNet_out), flat(bottleneck)]
    """
    def __init__(
        self,
        *,                       # force keyword args -> clearer
        dim: Tuple[int, int, int],
        bottleneck: bool = False,
        **kw
    ):
        # store image‑shape so we can use it later
        self.dim = dim
        super().__init__(dim=dim, **kw)  # torchcfm ctor

        self.bottleneck = bottleneck

        if bottleneck:
            # hook only the bottleneck layer
            self._feats, self._handles = [], []
            self._handles.append(
                self.middle_block.register_forward_hook(self._save_feat)
            )

        # -------- compute and publish output_dim --------
        C, H, W = dim
        base     = 1 + C * H * W         # [t, x_flat]

        if not bottleneck:
            self.output_dim = base + C * H * H         # + outptu of UNet
        else:
            cb   = self.channel_mult[-1] * self.num_channels
            res  = H // (2 ** (len(self.channel_mult) - 1))
            self.output_dim = base + C * H * H + cb * res * res

    def _save_feat(self, _mod, _inp, out):
        self._feats.append(out)

    def forward(self, inputs: torch.Tensor, y=None, *args, **kw):
        # unpack
        t    = inputs[:, :1]          # [B,1]
        flat = inputs[:, 1:]          # [B, C*H*W]
        B,   _ = flat.shape
        C, H, _ = self.dim
        x    = flat.view(B, C, H, H)  # [B,C,H,H]

        if self.bottleneck:
            self._feats.clear()

        # observables
        unet_out = super().forward(t.squeeze(1), x, y=y, *args, **kw)
        out_flat = unet_out.flatten(1)   # [B, C*H*W]

        # without coarse bottleneck
        if not self.bottleneck:
            return torch.cat([t, flat, out_flat], dim=1)

        # bottleneck-augmented i.e. append middle_block
        fb      = self._feats[0]             # [B, Cb, h, h]
        fb_flat = fb.flatten(1)              # [B, Cb*h*h]
        return torch.cat([t, flat, out_flat, fb_flat], dim=1)

    def __del__(self):
        for h in getattr(self, "_handles", []):
            h.remove()

class MultiUNetEncoder(nn.Module):
    """
    Wraps K identical UNetModelWrapper_encoder modules and concatenates their
    *UNet‑specific* features to [t, flat(x)].  
    If K = 1 it is equivalent to the single‑encoder case.
    """
    def __init__(
        self,
        *,
        K: int,
        dim: Tuple[int, int, int],
        bottleneck: bool = False,
        **unet_kwargs
    ):
        super().__init__()
        self.K = K
        self.base_dim = 1 + int(np.prod(dim))   # [t, x_flat]

        # --- create K independent encoders, all on CPU for now, helps with FSDP DDP etc
        self.encoders = nn.ModuleList([
            UNetModelWrapper_encoder(
                dim=dim,
                bottleneck=bottleneck,
                **unet_kwargs
            )
            for _ in range(K)
        ])

        # use the first to grab the per‑encoder extra size
        single_extra = self.encoders[0].output_dim - self.base_dim

        # publish our own output_dim so the rest of the code can read it
        self.output_dim = self.base_dim + K * single_extra

        C, H, _      = dim
        self.in_channels = C
        self.image_size  = H
        self.dim         = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z = [t, flat(x)]  – shape [B, 1 + C·H·W]
        Returns           – shape [B, output_dim]
        """
        t, flat = z[:, :1], z[:, 1:]
        extras  = []

        # Feed each encoder, cut away the duplicated [t, x] part
        for enc in self.encoders:
            full = enc(torch.cat([t, flat], dim=1))
            extras.append(full[:, self.base_dim:])   # only UNet-specific part
        return torch.cat([t, flat, *extras], dim=1)

class Decoder(nn.Module):
    def __init__(self, dim=(1, 28, 28)):
        super().__init__()
        C, H, W = dim
        self.input_dimension = C * H * W + 1
    
    def forward(self, tensor2d_x):
        return tensor2d_x[:, :self.input_dimension]

class Autoencoder_unet(nn.Module):
    """
    Auto‑encoder that can house K parallel UNet encoders.
    Set K (=num_encoders) in the Hydra config (default 1).
    """
    def __init__(
        self,
        *,
        dim: Tuple[int, int, int],
        num_encoders: int = 1,
        bottleneck: bool = False,
        # device: str = "cuda",
        **unet_kwargs
    ):
        super().__init__()

        if num_encoders == 1:
            self.encoder = UNetModelWrapper_encoder(
                dim        = dim,
                bottleneck = bottleneck,
                **unet_kwargs
            )#.to(device)
        else:
            self.encoder = MultiUNetEncoder(
                K          = num_encoders,
                dim        = dim,
                bottleneck = bottleneck,
                # device     = device,
                **unet_kwargs
            )

        self.decoder = Decoder(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

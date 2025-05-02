import torch
import torch.nn as nn
from typing import Optional
from torchcfm.models.unet import UNetModel
import torch.nn.functional as F

# UNetModel is the wrapper from torchcfm
class UNetModelWrapper_encoder(UNetModel):
    """
    UNet encoder with optional bottleneck augmentation.
      • bottleneck=False → [t, flat(x), flat(UNet_out)]
      • bottleneck=True  → [t, flat(x), flat(UNet_out), flat(bottleneck)]
    """
    def __init__(self, *args, bottleneck: bool = False, **kw):
        super().__init__(*args, **kw)
        self.bottleneck = bottleneck

        if bottleneck:
            # hook only the bottleneck layer
            self._feats, self._handles = [], []
            self._handles.append(
                self.middle_block.register_forward_hook(self._save_feat)
            )

    def _save_feat(self, _mod, _inp, out):
        self._feats.append(out)

    def forward(self, inputs: torch.Tensor, y=None, *args, **kw):
        # unpack
        t    = inputs[:, :1]             # [B,1]
        flat = inputs[:, 1:]             # [B, C*H*W]
        B, L = flat.shape
        C, H = self.in_channels, self.image_size
        x    = flat.view(B, C, H, H)     # [B,C,H,H]

        if self.bottleneck:
            self._feats.clear()

        # 1) run UNet to get standard output
        unet_out = super().forward(t.squeeze(1), x, y=y, *args, **kw)
        out_flat = unet_out.flatten(1)   # [B, C*H*W]

        # 2) basic branch: [t, x, UNet_out]
        if not self.bottleneck:
            return torch.cat([t, flat, out_flat], dim=1)

        # 3) bottleneck-augmented: also append middle_block
        fb      = self._feats[0]             # [B, Cb, h, h]
        fb_flat = fb.flatten(1)              # [B, Cb*h*h]
        return torch.cat([t, flat, out_flat, fb_flat], dim=1)

    def __del__(self):
        for h in getattr(self, "_handles", []):
            h.remove()

class Decoder(nn.Module):
    def __init__(self, dim=(1, 28, 28)):
        super().__init__()
        C, H, W = dim
        self.input_dimension = C * H * W + 1
    
    def forward(self, tensor2d_x):
        return tensor2d_x[:, :self.input_dimension]

class Autoencoder_unet(nn.Module):
    def __init__(self, dim, device='cuda', **unet_kwargs):
        super().__init__()
        self.encoder = UNetModelWrapper_encoder(
            dim=dim,
            **unet_kwargs
        ).to(device)
        self.decoder = Decoder(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
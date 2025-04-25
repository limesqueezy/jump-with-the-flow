import torch
import torch.nn as nn
from typing import Optional
from torchcfm.models.unet import UNetModel

# UNetModel is the wrapper from torchcfm
class UNetModelWrapper_encoder(UNetModel):
    """
    Thin subclass of torchcfm.models.unet.UNetModelWrapper that
    accepts combined [B, 1 + H*W] inputs: first element is timestep,
    rest is flattened image. Overrides only forward(); inherits
    all constructor logic and UNetModel functionality.
    """
    def forward(
        self,
        inputs: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        
        timesteps = inputs[:, 0]        # [B]
        flat      = inputs[:, 1:]       # [B, C*H*W]
        B, L      = flat.shape

        C = self.in_channels
        H = self.image_size
        W = self.image_size
        x = flat.view(B, C, H, W)       # [B, C, H, W]

        # Delegate to parent UNetModelWrapper.forward (which in turn calls UNetModel), y is not used since we dont use conditioning yet.
        out = super().forward(timesteps, x, y=y, *args, **kwargs)

        # Flatten output and concatenate with original inputs
        out_flat = out.view(B, L)
        return torch.hstack((inputs, out_flat))


class Decoder(nn.Module):
    def __init__(self, dim=(1, 28, 28)):
        super().__init__()
        C, H, W = dim
        self.input_dimension = C * H * W + 1
    
    def forward(self, tensor2d_x):
        return tensor2d_x[:, :self.input_dimension]

class Autoencoder_unet(nn.Module):
    def __init__(self, num_channels,num_res_blocks, dim, device='cuda'):
        super().__init__()
        self.encoder = UNetModelWrapper_encoder(
                dim=dim, 
                num_channels=num_channels,
                num_res_blocks=num_res_blocks
            ).to(device)
        self.decoder = Decoder(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
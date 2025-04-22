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
        # 1) Unpack timestep and flattened image
        timesteps = inputs[:, 0]
        flat      = inputs[:, 1:]
        B, L      = flat.shape
        side      = int(L ** 0.5)
        x         = flat.view(B, 1, side, side)

        # 2) Delegate to parent UNetModelWrapper.forward (which in turn calls UNetModel)
        out = super().forward(timesteps, x, y=y, *args, **kwargs)

        # 3) Flatten output and concatenate with original inputs
        out_flat = out.view(B, L)
        return torch.hstack((inputs, out_flat))


class Decoder(nn.Module):
    def __init__(self, dim=(1, 28, 28)):
        super().__init__()
        self.input_dimension = dim[-1]*dim[-2]+1
    
    def forward(self, tensor2d_x):
        return tensor2d_x[:,:self.input_dimension]

class Autoencoder_unet(nn.Module):
    def __init__(self, num_channels,num_res_blocks, dim, device='cuda'):
        super().__init__()
        self.encoder = UNetModelWrapper_encoder(
                dim=dim, 
                num_channels=num_channels,
                num_res_blocks=num_res_blocks
            ).to(device)
        self.decoder = Decoder(dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # def forward(self, tensor2d_x: torch.Tensor):
    #     tensor2d_x = self.encoder(tensor2d_x)
    #     return self.decoder(tensor2d_x)
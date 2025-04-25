from torchcfm.models.unet import UNetModel
from typing import Optional
import torch

# UNetModel is essentially the wrapper from the cfm codebase, we use the same thing to model the velocity field. TODO: Rename UNetWrapperKoopman to UNetKoopman
class UNetWrapperKoopman(UNetModel):
    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        *args,
        y: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        
        # Run UNet
        out = super().forward(t, x, y=y)
        B = out.shape[0]
        # Flatten all spatial-channel dimensions, prepend a 1
        flat = out.view(B, -1)
        ones = x.new_ones(B, 1)
        return torch.hstack((ones, flat))
import torch
import torch.nn as nn
import torch.nn.init as init

# OLD ONE TODO: Shouldn't we do torch no grad when we set init weights?
# class GenericOperator_state(nn.Module):
#     """ 
#     Generic operator with no particular parametrization
#     """
#     def __init__(self, operator_dim, init_std=1e-3):
#         super().__init__()
#         self.operator_dim = operator_dim
#         self.operator = nn.Parameter(torch.zeros((operator_dim,operator_dim)))
#         torch.nn.init.normal_(self.operator, mean=0.0, std=init_std)

class GenericOperator_state(nn.Module):
    def __init__(
        self,
        operator_dim: int,
        init_std: float = 1e-3,
    ):
        """
        operator_dim   – dimension of the square Koopman matrix
        identity_scale – how much of `I` to bake in (1.0 is full identity, 
                         <1.0 “softens” the prior)
        noise_std      – stddev of the N(0,1) noise before scaling
        """
        super().__init__()
        self.operator_dim = operator_dim
        self.identity_scale = 1e-2
        self.noise_std      = 1e-4

        # allocate the Parameter
        # (we'll fill it in with our custom init)
        self.operator = nn.Parameter(torch.empty(operator_dim, operator_dim))

        # do the one‐time init under no_grad
        with torch.no_grad():
            # 1) base identity
            init.eye_(self.operator)           # operator ← I
            self.operator.mul_(self.identity_scale) # operator ← I * identity_scale

            # 2) add small noise
            noise = torch.randn_like(self.operator) * self.noise_std
            self.operator.add_(noise)          # operator ← I*α + ε

        # now self.operator.requires_grad=True, gradients flow normally


    def forward(self, x):
        #tensor2d_x = torch.bmm(torch.vstack([self.operator.unsqueeze(0)]*tensor2d_x.shape[0]), tensor2d_x.reshape(tensor2d_x.shape[0],self.operator_dim,1))
        #print(tensor2d_x_trial.shape)
        x = x@self.operator
        #print(tensor2d_x.shape)
        return x.squeeze(-1)
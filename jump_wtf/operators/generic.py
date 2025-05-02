import torch
import torch.nn as nn
import torch.nn.init as init

# OLD ONE TODO: Shouldn't we do torch no grad when we set init weights?
class GenericOperator_state(nn.Module):
    """ 
    Generic operator with no particular parametrization
    """
    def __init__(self, operator_dim, init_std=1e-3):
        super().__init__()
        self.operator_dim = operator_dim
        self.operator = nn.Parameter(torch.zeros((operator_dim,operator_dim)))
        with torch.no_grad():
            torch.nn.init.normal_(self.operator, mean=0.0, std=init_std)

    def forward(self, x):
        #tensor2d_x = torch.bmm(torch.vstack([self.operator.unsqueeze(0)]*tensor2d_x.shape[0]), tensor2d_x.reshape(tensor2d_x.shape[0],self.operator_dim,1))
        #print(tensor2d_x_trial.shape)
        x = x@self.operator
        #print(tensor2d_x.shape)
        return x.squeeze(-1)
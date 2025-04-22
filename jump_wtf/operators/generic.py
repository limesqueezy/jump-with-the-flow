import torch
import torch.nn as nn

class GenericOperator_state(nn.Module):
    """ 
    Generic operator with no particular parametrization
    """
    def __init__(self, operator_dim):
        super().__init__()
        self.operator_dim = operator_dim
        self.operator = nn.Parameter(torch.zeros((operator_dim,operator_dim)))
        torch.nn.init.normal_(self.operator, mean=0.0, std=1e-3)
        
    def forward(self, tensor2d_x):
        #tensor2d_x = torch.bmm(torch.vstack([self.operator.unsqueeze(0)]*tensor2d_x.shape[0]), tensor2d_x.reshape(tensor2d_x.shape[0],self.operator_dim,1))
        #print(tensor2d_x_trial.shape)
        tensor2d_x = tensor2d_x@self.operator
        #print(tensor2d_x.shape)
        return tensor2d_x.squeeze(-1)
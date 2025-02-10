import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


''' 
Create a simple density estimator -- in this case, a simple MLP 
that maps out parameters \theta to a Gaussian distribution over 
the loss is then the negative log likleihood of N(f(theta)) at x
'''
class EmulatorNet(nn.Module):
    ''' Architecture configuration from Lueckmann 
        - output parameterize mean and covariance
        - hidden layer of 10 tanh units
    '''
    def __init__(self, theta_dim=1, hidden_dim=10):
        super(EmulatorNet, self).__init__()
        self.hidden = nn.Linear(theta_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = th.tanh(self.hidden(x))
        output = self.output(x)
        if len(x.shape) > 1:
            mean, var = output[:,0], output[:,1]
        else:
            mean, var = output[0], output[1]
        dist = Normal(mean, F.relu(var)+.0001)
        return dist
        
import torch
import torch.nn as nn
import numpy as np
from pinns.rff import GaussianEncoding
from collections import OrderedDict

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

# Function to enforce hard constraints
def hard_constraint(x, y):
    '''
    Enforces output y=[V,u] to have V(0)=0 and V_dot(0)=0
    
    x_example = [0; 2; 0]          # Shape (3, 1)
    y_example = [4, 5; 6, 7; 8, 9] # Shape (3, 2)
    x_example * y_example = [0, 0; 12, 14; 0, 0]
    '''
    V = x * y[:, 0:1]  # Element-wise multiplication to apply the constraint
    u = y[:, 1:2]
    # Concatenate V and u along the second dimension
    constrained_y = torch.cat((V, u), dim=1)
    return constrained_y

# Define the ODE function for the Lyapunov constraints
def integrator_ode(pred, x):
    '''
    System considered: INTEGRATOR
    - system: dxdt = u
    - input:  u to be learned  
    '''
    dxdt =  pred[:,1] #+ 0.6*sin(x[:,1])
    dVdx = jacobian(pred, x, i=0,j=0) 
    return dVdx*dxdt
    
class PINN(nn.Module):

    def __init__(self, layers, activation_function, hard_constraint_fn=None, ff=False, sigma=None):
        super(PINN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = activation_function 
        self.ff = ff      
        layer_list = list()
        if self.ff:
            if sigma is None:
                raise ValueError("If Random Fourier Features embedding is on, then a sigma must be specified")
            self.encoding = GaussianEncoding(sigma=sigma, input_size=layers[0], encoded_size=layers[0])
            layers[0] *= 2

        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        self.hard_constraint_fn = hard_constraint_fn
        
    def forward(self, x):
        try:
            if self.ff:
                x = self.encoding(x)
        except:
            pass
        output = self.layers(x)
        #y_cpu = output.cpu().detach().numpy()
        #for i in range(y_cpu.shape[0]):
        #    print(f"output without hard_constr:{y_cpu[i][0]}")
        
        if self.hard_constraint_fn is not None:
            output = self.hard_constraint_fn(x, output)
        return output

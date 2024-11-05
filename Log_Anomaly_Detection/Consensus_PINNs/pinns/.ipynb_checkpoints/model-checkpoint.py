import torch
import torch.nn as nn
import numpy as np
from pinns.rff import GaussianEncoding
from collections import OrderedDict

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

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
            #V = output[:, 0:1]
            #u = output[:, 1:]
            #V_constrained = self.hard_constraint_fn(x, V)
            #output = torch.cat((V_constrained, u), dim=1)
        
        # Ensure V(x) > 0 by applying softplus activation function (commented bc im trying with loss_positive)
        #output = torch.nn.functional.softplus(output)
        return output

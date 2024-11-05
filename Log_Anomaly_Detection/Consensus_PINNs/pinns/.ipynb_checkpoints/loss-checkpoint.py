import torch
import torch.nn as nn
from pinns.gradient import jacobian
import numpy as np
import torch.nn.functional as F
'''
def compute_loss(normalization, samples, model, ode_fn, Q, positive_weight=1.0, derivative_weight=1.0, properness_weight=1.0):
    samples.requires_grad = True
    predictions = model(samples)
    if normalization is True:
        # lie(xi)
        lie,theta_min, D_theta = ode_fn(predictions, samples)
        radius = torch.norm(samples+(theta_min/D_theta), p=2, dim=1) # 0.5 = theta_min / Dtheta
    else: 
        lie, dVdx1, dVdx2 = ode_fn(predictions, samples)
        radius = torch.norm(samples, p=2, dim=1)
    # Ensure V(xi) > 0
    #loss_positive = torch.mean(torch.abs(nn.functional.gelu(-predictions[:,0])))*positive_weight
    #loss_positive = torch.mean(torch.relu(-predictions[:,0])**2) * positive_weight
    loss_positive = torch.max(torch.relu(-predictions[:,0])**2) * positive_weight
    # Constraint dV/dx < -Q
    #loss_derivative = torch.mean(torch.relu(lie + Q)**2) * derivative_weight
    #loss_derivative = torch.max(torch.relu(lie + Q)**2) * derivative_weight
    loss_derivative = torch.mean(nn.functional.gelu(lie + Q)**2) * derivative_weight
    #loss_derivative = torch.max(nn.functional.gelu(lie + Q)**2) * derivative_weight
    # Properness: V(x) should be large for large x
    #loss_properness = torch.mean(nn.functional.gelu(radius - predictions[:,0])**2)*properness_weight
    loss_properness = torch.mean(torch.relu(radius - predictions[:,0])**2)*properness_weight
    #x_0_tensor = torch.tensor([[0.5, 0.5]], dtype=torch.float32, requires_grad=True).to('cuda:0')
    #output_0 = model(x_0_tensor)
    #Lyapunov_risk = (F.relu(-predictions[:,0])+ 1.5*F.relu(lie+Q)).mean()

    #return Lyapunov_risk,torch.tensor(0),torch.tensor(0)
    return loss_positive, loss_derivative, loss_properness

def compute_loss(normalization, samples, model, ode_fn, Q, positive_weight=1.0, derivative_weight, properness_weight):
    samples.requires_grad = True
    predictions = model(samples)
    V = predictions[:, 0]
    
    # Compute the Lie derivative of V (i.e., dV/dt)
    lie, dVdx1, dVdx2 = ode_fn(predictions, samples)
    
    # Radius for distance-based constraints
    radius = torch.norm(samples, p=2, dim=1)
    
    # Ensure V(theta, omega) > 0 for all (theta, omega) != (0, 0)
    loss_positive = torch.max(torch.relu(radius-V)**2) * positive_weight
    
    # Constraint dV/dt < -Q for far from zero, dV/dt < 0 for close to zero
    # Applying different regions for constraints based on the radius
    
    loss_derivative_close = torch.max(torch.relu(lie*radius)**2) * derivative_weight
    loss_derivative_far = torch.max(torch.relu((lie + Q)*radius)**2) * properness_weight
    return loss_positive, loss_derivative_close, loss_derivative_far
'''
def integrator_loss(check_alpha, samples, model, ode_fn, lie_params, positive_weight, derivative_weight, properness_weight):
    samples.requires_grad = True
    predictions = model(samples)
    V = predictions[:, 0]
    # Compute the Lie derivative of V (i.e., dV/dt)
    lie = ode_fn(predictions, samples)
    
    # Radius for distance-based constraints
    radius = torch.norm(samples, p=2, dim=1)
    
    # Ensure V(theta, omega) > 0 for all (theta, omega) != (0, 0)
    loss_positive = torch.max(torch.relu(radius-V)**2) * positive_weight

    # Ensure dVdt < 0
    loss_derivative_neg = torch.max(torch.relu(lie*radius)**2) * derivative_weight
    
    if not check_alpha:
        # Constraint dV/dt < -Q for far from zero, dV/dt < 0 for close to zero
        Q = lie_params
        loss_derivative_constr = torch.max(torch.relu((lie + Q)*radius)**2) * properness_weight
    else:
        alpha = lie_params[0]
        delta = lie_params[1]
        c = lie_params[2]
        V_alpha = torch.zeros_like(V)
        for i, vi in enumerate(V):
            if vi<0:
                vi = float(vi)
                vi_1 = vi**alpha
                vi_alpha = - np.sqrt(np.square(vi_1.real) + np.square(vi_1.imag)) 
            else:
                vi_alpha = vi**alpha  
            V_alpha[i] = vi_alpha
            
        constraint = -c*V_alpha+delta
        #loss_derivative_constr = torch.tensor(0)
        loss_derivative_constr = torch.max(torch.relu((lie - constraint))**2) * properness_weight
    #print(loss_positive, loss_derivative_neg, loss_derivative_constr)
    return loss_positive, loss_derivative_neg, loss_derivative_constr

def inv_pendulum_loss(check_alpha, samples, model, ode_fn, lie_params, positive_weight, derivative_weight, properness_weight):
    samples.requires_grad = True
    predictions = model(samples)
    V = predictions[:, 0]
    # Compute the Lie derivative of V (i.e., dV/dt)
    lie, dVdx1, dVdx2 = ode_fn(predictions, samples)
    
    # Radius for distance-based constraints
    radius = torch.norm(samples, p=2, dim=1)
    
    # Ensure V(theta, omega) > 0 for all (theta, omega) != (0, 0)
    loss_positive = torch.max(torch.relu(radius-V)**2) * positive_weight

    # Ensure dVdt < 0
    loss_derivative_neg = torch.max(torch.relu(lie*radius)**2) * derivative_weight
    
    if not check_alpha:
        # Constraint dV/dt < -Q for far from zero, dV/dt < 0 for close to zero
        Q = lie_params
        loss_derivative_constr = torch.max(torch.relu((lie + Q)*radius)**2) * properness_weight
    else:
        alpha = lie_params[0]
        delta = lie_params[1]
        c = lie_params[2]
        #constraint = -c*V**alpha+delta
        V_alpha = torch.zeros_like(V)
        for i, vi in enumerate(V):
            if vi<0:
                vi = float(vi)
                vi_1 = vi**alpha
                vi_alpha = - np.sqrt(np.square(vi_1.real) + np.square(vi_1.imag)) 
            else:
                vi_alpha = vi**alpha  
            V_alpha[i] = vi_alpha
            
        constraint = -c*V_alpha+delta
        #loss_derivative_constr = torch.tensor(0)
        loss_derivative_constr = torch.max(torch.relu((lie - constraint))**2) * properness_weight
    #print(loss_positive, loss_derivative_neg, loss_derivative_constr)
    return loss_positive, loss_derivative_neg, loss_derivative_constr
    
def compute_loss(dynamics, check_alpha, samples, model, ode_fn, lie_params, positive_weight=1.0, derivative_weight=1.0, properness_weight=1.0):
    if dynamics=="Inv_pend_PINN":
        return inv_pendulum_loss(check_alpha, samples, model, ode_fn, lie_params, positive_weight, derivative_weight, properness_weight)
    elif dynamics=="Integrator_PINN":
        return integrator_loss(check_alpha, samples, model, ode_fn, lie_params, positive_weight, derivative_weight, properness_weight)
    else:
        raise ValueError(f'The dynamics {dynamics} is not recognized.')
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

# AUXILIARY FUNCTIONS
def generate_random_state_ranges(M, overall_range=(-6, 6)):
    state_ranges = []
    for _ in range(M):
        min_val = np.random.uniform(overall_range[0], overall_range[1])
        max_val = np.random.uniform(min_val, overall_range[1])
        state_ranges.append((min_val, max_val))
    return state_ranges
    
def saturation(x, delta):
    return np.clip(x, -delta, delta)
    
def disturbance(param, x, t):
    return param * np.sin(t)

def initialize_integrator_states(N, M, state_ranges):
    initial_states = np.zeros((N, M))
    for m in range(M):
        min_val, max_val = state_ranges[m]
        initial_states[:, m] = np.random.uniform(min_val, max_val, N)
    return initial_states

def compute_T(model, initial_states, alpha, c, theta):
    N, M = initial_states.shape
    T = []
    for m in range(M):
        # Extract the m-th column of initial_states matrix and reshape to (N, 1)
        column = initial_states[:, m].reshape(-1, 1)
        column_tensor = torch.tensor(column, dtype=torch.float32).to('cuda:0')
        
        # Find the m-th initial value of Lyapunov function V(x0)
        outputs = model(column_tensor)
        V_initial = outputs[:, 0].cpu().detach().numpy()

        # Find the m-th V(x0)**(1-alpha)
        V_alpha = np.zeros_like(V_initial)
        for i, vi in enumerate(V_initial):
            if vi<0:
                vi = float(vi)
                vi_1 = vi**(1-alpha)
                vi_alpha = - np.sqrt(np.square(vi_1.real) + np.square(vi_1.imag)) 
            else:
                vi_alpha = vi**(1-alpha) 
            V_alpha[i] = vi_alpha
            
        # Compute T for the m-th column
        T_m = (V_alpha) / (c * theta * (1 - alpha))
        T.append(T_m)
    return np.array(T)

    
def leader_integrator_control(state, model):
    input_tensor = torch.tensor(np.array([state]).reshape(-1,1), dtype=torch.float32).to('cuda:0')
    control = model(input_tensor)[:, 1].item()
    return control

def integrator_protocol(A, X, k1, k2, alpha, delta1, model):
    N, M = X.shape
    U = np.zeros((N, M))
    for i in range(N):
        for m in range(M):
            vi = sum(A[i, j] * (X[i, m] - X[j, m]) for j in range(N))
            if vi < 0:
                vi = float(vi)
                vi_1 = vi**alpha
                vi_alpha = - np.sqrt(np.square(vi_1.real) + np.square(vi_1.imag))
            else:
                vi_alpha = vi**alpha
            ui = -saturation(k1 * vi_alpha, delta1) - k2 * np.tanh(vi)
            U[i, m] = ui
    return U
'''
def integrator_network_dynamics(t, state_flat, A, k1, k2, alpha, delta1, model, dist):
    N, M = A.shape[0], state_flat.size // A.shape[0]
    state = state_flat.reshape(N, M)
    dxdt = np.zeros((N, M))
    U = integrator_protocol(A, state, k1, k2, alpha, delta1, model)
    for i in range(N):
        for m in range(M):
            state_value = state[i, m]
            dxdt[i, m] = saturation(U[i, m], delta1)
    return dxdt.flatten()

def simulate_network(N, M, T, dt, A, k1, k2, alpha, delta1, initial_states, model, dist=None):
    t_span = (0, T*dt)
    t_eval = np.linspace(t_span[0], t_span[1], T)
    initial_states_flat = initial_states.flatten()
    
    sol = solve_ivp(integrator_network_dynamics, t_span, initial_states_flat, args=(A, k1, k2, alpha, delta1, model, dist), t_eval=t_eval, method='RK23')
    x_trajectory = sol.y.T.reshape(-1, N, M)
    return sol.y[:, -1].reshape(N, M), x_trajectory, sol.t
'''

def integrator_network_dynamics(t, state_flat, A, k1, k2, alpha, delta1, model, dist, T):
    N, M = A.shape[0], state_flat.size // A.shape[0]
    state = state_flat.reshape(N, M)
    dxdt = np.zeros((N, M))
    
    U = integrator_protocol(A, state, k1, k2, alpha, delta1, model)
    for i in range(N):
        for m in range(M):
            if t <= T[m]:
                dxdt[i, m] = saturation(U[i, m], delta1)#U[i,m]
            else:
                dxdt[i, m] = 0  # No evolution if t exceeds T[m]
    
    return dxdt.flatten()

def simulate_network(N, M, T, dt, A, k1, k2, alpha, delta1, initial_states, model, dist=None):
    T_max = np.max(T)
    t_eval = np.linspace(0, T_max, int(T_max / dt))
    
    initial_states_flat = initial_states.flatten()
    
    sol = solve_ivp(integrator_network_dynamics, [0, T_max], initial_states_flat, args=(A, k1, k2, alpha, delta1, model, dist, T), t_eval=t_eval, method='RK23')
    x_trajectory = sol.y.T.reshape(-1, N, M)
    
    final_state = np.zeros((N, M))
    for m in range(M):
        time_index = np.searchsorted(t_eval, T[m])
        final_state[:, m] = x_trajectory[time_index, :, m]
    
    return final_state, x_trajectory, sol.t


def plot_convergence_M_states(x_trajectory, time):
    plt.figure(figsize=(12, 6))
    N, M = x_trajectory.shape[1], x_trajectory.shape[2]

    consensus_values = np.mean(x_trajectory[-1, :, :], axis=0)

    for i in range(N):
        for m in range(M):
            plt.plot(time, x_trajectory[:, i, m])
    
    for m in range(M):
        plt.plot(time, np.ones_like(time) * consensus_values[m], linestyle='--', label=f'Consensus $x_{m+1}(T)$={consensus_values[m]:.8f}')
    
    plt.title('Convergence of Integrator States')
    plt.xlabel('Time Steps')
    plt.ylabel('$x$')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_integrator_M_controls(x_trajectory, time):
    """
    Plot the control evolutions for the agents in the case of an integrator.

    Parameters:
    x_trajectory (np.ndarray): The trajectory of states over time (T x N x M).
    time (np.ndarray): The time values corresponding to the trajectory.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    N, M = x_trajectory.shape[1], x_trajectory.shape[2]  # Number of agents and integrators

    for m in range(M):
        plt.figure(figsize=(12, 6))
        for i in range(N):
            u = np.gradient(x_trajectory[:, i, m], time)
            plt.plot(time, u)#, label=f'Control $u_{{{i},{m}}}(T)$={u[-1]:.8f}')
        plt.title(f'Control Evolutions for Integrator {m+1}')
        plt.xlabel('Time [s]')
        plt.ylabel('Control Input $u$')
        #plt.legend(loc='best')
        plt.grid()
        plt.show()

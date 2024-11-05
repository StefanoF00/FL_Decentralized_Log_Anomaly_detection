from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

G = 9.81     # gravity
L = 0.5      # length of the pole
m = 0.15     # ball mass
b = 0.1      # friction
J = m * L**2 # Inertia 

def saturation(x, delta):
    return np.clip(x, -delta, delta)
    
# STATE INITIALIZATION FUNCTIONS
def initialize_inv_pendulum_states(N, M, state_ranges):
    initial_states = np.zeros((N, M, 2)) 
    for m in range(M):
        theta_min, theta_max = state_ranges[m][0]
        omega_min, omega_max = state_ranges[m][1]
        initial_states[:, m, 0] = np.random.uniform(theta_min, theta_max, N)-np.pi
        initial_states[:, m, 1] = np.random.uniform(omega_min, omega_max, N)
    return initial_states


# LEADER CONTROLS FUNCTIONS
def leader_inv_pendulum_control(theta, omega, model):
    input_tensor = torch.tensor([[theta, omega]], dtype=torch.float32).to('cuda:0')
    control = model(input_tensor)[:, 1].item()
    return control

# CONTROL PROTOCOLS FUNCTIONS
def inv_pendulum_protocol(A, X, k1, k2, alpha, delta1, model):
    N, M = X.shape[:2]
    U = np.zeros((N, M))
    for m in range(M):
        for i in range(N):
            # CASE 1 (No leader- follower structure)
            
            vi = sum(A[i, j] * (X[i, m, 0] - X[j, m, 0]) for j in range(N))
            if vi < 0:
                vi = float(vi)
                vi_1 = vi**alpha
                vi_alpha = - np.sqrt(np.square(vi_1.real) + np.square(vi_1.imag))
            else:
                vi_alpha = vi**alpha
            ui = -saturation(k1 * vi_alpha, delta1) - k2 * np.tanh(vi)
            U[i, m] = ui
            # CASE 2 (Leader-Follower structure, convergence with learned control)
            '''
            if i == 0:
                # Leader node
                U[i,m] = leader_inv_pendulum_control(X[i, m, 0], X[i, m, 1], model)
            else:
                vi = sum(A[i, j] * (X[i, m, 0] - X[j, m, 0]) for j in range(N))
                if vi<0:
                    vi = float(vi)
                    vi_1 = vi**alpha
                    vi_alpha = - np.sqrt(np.square(vi_1.real) + np.square(vi_1.imag))
                    
                else:
                    vi_alpha = vi**alpha
                ui = -saturation(k1 * vi_alpha, delta1) - k2 * np.tanh(vi)
                U[i,m] = ui
            '''
    return U
    
# DYNAMICAL SYSTEMS FUNCTIONS
def inv_pendulum_network_dynamics(t, state_flat, N, M, A, k1, k2, alpha, delta1, model):
    state = state_flat.reshape(N, M, 2)
    dxdt = np.zeros((N, M, 2))
    U = inv_pendulum_protocol(A, state, k1, k2, alpha, delta1, model)
    
    for i in range(N):
        for m in range(M):
            theta = state[i, m, 0]
            omega = state[i, m, 1]
            dtheta_dt = omega
            domega_dt = (-m * G * L * np.sin(theta + np.pi) - b * omega + U[i, m]) / J
            
            dxdt[i, m, 0] = dtheta_dt
            dxdt[i, m, 1] = domega_dt
            
    return dxdt.flatten()

# SIMULATION FUNCTION
def simulate_network(N, M, T, dt, A, k1, k2, alpha, delta1, initial_states, model, dist=None):
    t_span = (0, T*dt)
    t_eval = np.linspace(t_span[0], t_span[1], T)
    initial_states_flat = initial_states.flatten()
    
    sol = solve_ivp(inv_pendulum_network_dynamics, t_span, initial_states_flat, args=(N, M, A, k1, k2, alpha, delta1, model), t_eval=t_eval, method='RK23')
    
    x_trajectory = sol.y.T.reshape(-1, N, M, 2)
    for i in range(N):
        for m in range(M):
            x_trajectory[:, i, m, 0] += np.pi
    return sol.y[:, -1].reshape(N, M, 2), x_trajectory, sol.t

# PLOTTING FUNCTIONS
def plot_convergence_theta(x_trajectory, time):
    plt.figure(figsize=(12, 6))
    N, M = x_trajectory.shape[1], x_trajectory.shape[2]

    for m in range(M):
        for i in range(N):
            plt.plot(time, x_trajectory[:, i, m, 0])
    consensus_values = np.mean(x_trajectory[-1, :, :, 0], axis=0)
    for m in range(M):
        plt.plot(time, np.ones_like(time) * consensus_values[m], linestyle='--', label=f'Consensus $\\theta_{{{m}}}(T)$={consensus_values[m]/np.pi:.8f}pi')
    
    plt.title('Convergence of $\\theta$')
    plt.xlabel('Time Steps')
    plt.ylabel('$\\theta$ [rad]')
    plt.legend(loc='best')
    plt.grid()
    # os.makedirs('fig', exist_ok=True)
    #plt.savefig(f'fig/{protocol_name}_theta.png')
    plt.show()

def plot_convergence_omega(x_trajectory, time):
    plt.figure(figsize=(12, 6))
    N, M = x_trajectory.shape[1], x_trajectory.shape[2]

    for m in range(M):
        for i in range(N):
            plt.plot(time, x_trajectory[:, i, m, 1])
    consensus_values = np.mean(x_trajectory[-1, :, :, 1], axis=0)
    for m in range(M):
        plt.plot(time, np.ones_like(time) * consensus_values[m], linestyle='--', label=f'Consensus $\\omega_{{{m}}}(T)$={consensus_values[m]:.8f}')
    
    plt.title('Convergence of $\\omega$')
    plt.xlabel('Time Steps')
    plt.ylabel('$\\omega$ [rad/s]')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def plot_inv_pendulum_control(x_trajectory, time):
    plt.figure(figsize=(12, 6))
    N, M = x_trajectory.shape[1], x_trajectory.shape[2]

    for m in range(M):
        plt.figure(figsize=(12, 6))
        for i in range(N):
            omega = x_trajectory[:, i, m, 1]
            u = np.gradient(omega, time) * J + b * omega + m * G * L * np.sin(x_trajectory[:, i, m, 0] + np.pi)
            plt.plot(time, u)
        consensus_values = np.mean(np.gradient(x_trajectory[-1, :, m, 1], time[-1]) * J + b * x_trajectory[-1, :, m, 1] + m * G * L * np.sin(x_trajectory[-1, :, m, 0] + np.pi), axis=0)
        
        plt.plot(time, np.ones_like(time) * consensus_values, linestyle='--', label=f'Consensus Control $u_{{{m}}}(T)$={consensus_values:.8f}')
        
        plt.title(f'Control Evolutions for Inverse pendulum {m+1}')
        plt.xlabel('Time [s]')
        plt.ylabel('Control Input $u$')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

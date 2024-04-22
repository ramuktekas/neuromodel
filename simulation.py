import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import detrend, butter, filtfilt

# Define all necessary functions at the beginning
def detrend(ts):
    return scipy.signal.detrend(ts)
def sigmoid_activation(input, k):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-k * input))

def wilsoncowan_dxdt(x, s_input, tau, bias, W, k, C):
    """Wilson-Cowan differential equation."""
    dx_input = C * np.dot(W, x) + bias + s_input
    return (-x + sigmoid_activation(dx_input, k)) / tau

def wilsoncowan_RK2(tau, bias, W, k, s, C, tspan, dt, burn):
    """Wilson-Cowan model simulation using 2nd-order Runge-Kutta method."""
    nrois = W.shape[0]
    time = np.arange(0, tspan + dt, dt)
    s_input = np.full((nrois, len(time)), s) + np.random.randn(nrois, len(time))
    x = np.zeros((nrois, len(time)))
    x[:, 0] = np.random.randn(nrois)
    for t in range(len(time) - 1):
        k1 = (dt / 2) * wilsoncowan_dxdt(x[:, t], s_input[:, t], tau, bias, W, k, C)
        k2 = dt * wilsoncowan_dxdt(x[:, t] + k1, s_input[:, t], tau, bias, W, k, C)
        x[:, t + 1] = x[:, t] + k2
    return x[:, int(burn / dt):], time[int(burn / dt):], s_input

def bw(r, T, dt):
    """Balloon-Windkessel model for BOLD signal simulation."""
    params = {'taus': 0.65, 'tauf': 0.41, 'tauo': 0.98, 'alpha': 0.32, 'Eo': 0.34, 'vo': 0.02}
    k1, k2, k3 = 7 * params['Eo'], 2, 2 * params['Eo'] - 0.2
    itaus, itauf, itauo, ialpha = 1 / params['taus'], 1 / params['tauf'], 1 / params['tauo'], 1 / params['alpha']
    nrois, n_t = r.shape[0], int(T / dt) + 1
    x = np.zeros((n_t, 4, nrois))
    x[0, :, :] = np.tile([0, 1, 1, 1], (nrois, 1)).T
    for n in range(n_t - 1):
        x[n + 1, 0, :] = x[n, 0, :] + dt * (r[:, n] - itaus * x[n, 0, :] - itauf * (x[n, 1, :] - 1))
        x[n + 1, 1, :] = x[n, 1, :] + dt * x[n, 0, :]
        x[n + 1, 2, :] = x[n, 2, :] + dt * itauo * (x[n, 1, :] - x[n, 2, :] ** ialpha)
        x[n + 1, 3, :] = x[n, 3, :] + dt * itauo * ((x[n, 1, :] * (1 - (1 - params['Eo']) ** (1 / x[n, 1, :])) / params['Eo']) - (x[n, 2, :] ** ialpha * x[n, 3, :] / x[n, 2, :]))
    return 100 / params['Eo'] * params['vo'] * (k1 *(1 - x[:, 3, :] / x[:, 2, :]) + k3 * (1 - x[:, 2, :])).squeeze().T

# Load the connectivity matrix
connectivity_matrix_path = '/Users/satrokommos/Documents/modelling_wilsoncowan/averageConnectivity_Fpt.mat'
with h5py.File(connectivity_matrix_path, 'r') as f:
    W = np.array(f['/Fpt'])
    W[np.isnan(W)] = 0  # Replace NaN values with 0


results_dict = {}

specific_rois = [24, 25, 107] #
# Simulation parameters - define or adjust these according to your needs
# Simulation parameters
tau, bias, k, s, C, tspan, dt, burn = 1.0, -3, 0.5, 0, 0.1, 1000, 0.01, 100
T_BOLD, dt_BOLD = tspan - burn, 0.01

# Perform the simulation...
x, time, s_input = wilsoncowan_RK2(tau, bias, W, k, s, C, tspan, dt, burn)
bold_signal = bw(x, T_BOLD, dt_BOLD)

# Ensure the time and bold_signal arrays start after the burn period
start_index = int(burn / dt)
time_adjusted = time[start_index:]
bold_signal_adjusted = bold_signal[:, start_index:]

# Plotting the BOLD signal for specific ROIs
plt.figure(figsize=(12, 6))
for idx, roi in enumerate(specific_rois):
    plt.plot(time_adjusted, bold_signal_adjusted[roi], label=f'ROI {roi + 1}')  # +1 to match MATLAB 1-based indexing

plt.title('Simulated BOLD Signal')
plt.xlabel('Time (s)')
plt.ylabel('BOLD Signal')
plt.legend()
plt.show()





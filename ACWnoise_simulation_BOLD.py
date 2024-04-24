import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import detrend, butter, filtfilt, fftconvolve, lfilter

# Define all necessary functions at the beginning
def sigmoid_activation(input, k):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-k * input))

def wilsoncowan_dxdt(x, s_input, tau, bias, W, k, C):
    """Wilson-Cowan differential equation."""
    dx_input = C * np.dot(W, x) + bias + s_input
    return (-x + sigmoid_activation(dx_input, k)) / tau

def wilsoncowan_RK2(tau, bias, W, k, C, tspan, dt, burn, acw, amplitude, n_lags):
    """Wilson-Cowan model simulation using 2nd-order Runge-Kutta method with custom noise as input."""
    nrois = W.shape[0]
    time = np.arange(0, tspan + dt, dt)
    acf = desired_acf(n_lags, acw)
    s_input = generate_colored_noise(acf, len(time) * nrois, amplitude).reshape(nrois, len(time))
    x = np.zeros((nrois, len(time)))
    x[:, 0] = np.random.randn(nrois)
    for t in range(len(time) - 1):
        k1 = (dt / 2) * wilsoncowan_dxdt(x[:, t], s_input[:, t], tau, bias, W, k, C)
        k2 = dt * wilsoncowan_dxdt(x[:, t] + k1, s_input[:, t], tau, bias, W, k, C)
        x[:, t + 1] = x[:, t] + k2
    return x[:, int(burn / dt):], time[int(burn / dt):], s_input[:, int(burn / dt):]

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

def desired_acf(lag, acw):
    """Generate a desired ACF that decays exponentially and truncates at a specific lag."""
    acf = np.exp(-np.arange(lag) / acw)
    acf[acw:] = 0  # Truncate after specified acw
    return acf

def generate_colored_noise(acf, n_samples, amplitude=1.0):
    """Generate noise with a specified autocorrelation function."""
    # Generate white noise
    white_noise = np.random.normal(0, 1, n_samples)

    # Filter white noise with the ACF as FIR filter coefficients
    colored_noise = lfilter(acf, [1.0], white_noise)

    return amplitude * colored_noise

# Load the connectivity matrix
connectivity_matrix_path = '/Users/satrokommos/Documents/modelling_wilsoncowan/averageConnectivity_Fpt.mat'
with h5py.File(connectivity_matrix_path, 'r') as f:
    W = np.array(f['/Fpt'])
    W[np.isnan(W)] = 0  # Replace NaN values with 0

# Simulation parameters
tau, bias, k, C, tspan, dt, burn = 1.0, -3, 0.5, 0.1, 1000, 0.01, 0
T_BOLD, dt_BOLD = tspan - burn, 0.01
acw, amplitude, n_lags = 100, 1.0, 1000  # Custom noise parameters

# Perform the simulation with custom noise...
x, time, s_input = wilsoncowan_RK2(tau, bias, W, k, C, tspan, dt, burn, acw, amplitude, n_lags)
bold_signal = bw(x, T_BOLD, dt_BOLD)

# Plotting the BOLD signal for specific ROIs
plt.figure(figsize=(12, 6))
specific_rois = [24, 25, 107]
for idx, roi in enumerate(specific_rois):
    plt.plot(time, bold_signal[roi], label=f'ROI {roi + 1}')  # +1 to match MATLAB 1-based indexing

plt.title('Simulated BOLD Signal with Custom Noise')
plt.xlabel('Time (s)')
plt.ylabel('BOLD Signal')
plt.legend()
plt.show()


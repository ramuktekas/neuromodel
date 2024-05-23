import numpy as np
import matplotlib.pyplot as plt
import h5py
from PyIF import te_compute as te
import pandas as pd
from scipy.signal import butter, filtfilt
import scipy.signal
from statsmodels.tsa.stattools import acf
from sklearn.utils import shuffle
import random

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

def discretize_signal(signal, dt, interval=1.0):
    """Discretize signals into one-second intervals."""
    steps_per_interval = int(interval / dt)
    num_intervals = signal.shape[1] // steps_per_interval
    return np.array([signal[:, i * steps_per_interval:(i + 1) * steps_per_interval].mean(axis=1) for i in range(num_intervals)]).T

def calculate_te(source, target, m):
    """Calculate Transfer Entropy with varying tau values."""
    # Ensure both source and target are arrays and not empty or scalar
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()
    if source.size == 0 or target.size == 0:
        raise ValueError("Source or target is empty.")
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Source or target is not 1-dimensional.")
    
    min_length = min(len(source), len(target))
    source = source[:min_length]
    target = target[:min_length]
    tau_values = np.arange(1, 2)  # Only tau=1 considered here; adjust if needed
    te_results = [te.te_compute(source, target, m, tau) for tau in tau_values]
    return te_results

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

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
    W[np.isnan(W)] = 1.5  # Replace NaN values with 0

#Main simulation and analysis


#Prepare to accumulate results
te_values_dict = {}

def apply_bandpass(data, lowcut=0.02, highcut=0.1, fs=1.0, order=5):
    """Apply bandpass filtering to a given time series."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_acw(ts):
    """Calculate the Autocorrelation Window (ACW) for a given time series."""
    acw_func = acf(ts, nlags=len(ts)-1, fft=True)
    acw_0 = np.argmax(acw_func <= 0)
    return acw_0

def get_acw_time_series(time_series, window_size=150, step_size=1):
    """Generate a time series of ACW values for a given signal."""
    detrended_ts = detrend(time_series)
    filtered_ts = apply_bandpass(detrended_ts)
    acw_series = [calculate_acw(filtered_ts[i:i+window_size]) for i in range(0, len(filtered_ts) - window_size + 1, step_size)]
    return acw_series


def markov_block_bootstrap(time_series, sampling_rate=1.0):
    # Calculate block length using the first zero-crossing of the autocorrelation function
    acw_func = acf(time_series, nlags=len(time_series) - 1, fft=True)
    block_length = np.argmax(acw_func <= 0) / sampling_rate

    # Ensure block length is at least 1
    block_length = max(int(block_length), 1)

    num_blocks = int(np.ceil(len(time_series) / block_length))
    blocks = [time_series[i * block_length:(i + 1) * block_length] for i in range(num_blocks)]

    # Shuffle the blocks
    shuffled_blocks = shuffle(blocks, random_state=None)

    # Flatten the list of shuffled blocks into a single time series
    shuffled_time_series = [item for sublist in shuffled_blocks for item in sublist]
    return shuffled_time_series


def philipp_shuffle(time_series, sampling_rate=1.0):
    # Calculate block length using the first zero-crossing of the autocorrelation function
    acw_func = acf(time_series, nlags=len(time_series) - 1, fft=True)
    block_length = np.argmax(acw_func <= 0) / sampling_rate

    # Ensure block length is at least 1
    block_length = max(int(block_length), 1)

    num_blocks = int(np.ceil(len(time_series) / block_length))
    blocks = [time_series[i * block_length:(i + 1) * block_length] for i in range(num_blocks)]

    # Shuffle within each block
    shuffled_blocks = [np.random.permutation(block) for block in blocks]

    # Flatten the list of shuffled blocks into a single time series
    shuffled_time_series = [item for sublist in shuffled_blocks for item in sublist]
    return shuffled_time_series


def p_value(input_ts, shuffle_ts, te_values):
    # Initialize a dictionary to store the counts of shuffled TE values greater than original TE values for each Tau
    greater_count = {tau: 0 for tau in range(len(te_values))}

    # Loop through each shuffled time series
    for ts in shuffle_ts:
        shuffle_te = calculate_te(input_ts, ts, 3)

        # Increment the count for each Tau if the shuffled TE is greater than or equal to the original TE
        for tau, te_value in enumerate(shuffle_te):
            if te_value >= te_values[tau]:
                greater_count[tau] += 1

    # Calculate p-values for each Tau
    p_values = {tau: greater_count[tau] / len(shuffle_ts) for tau in range(len(te_values))}

    return p_values


def run_simulation_and_analysis(ACW=False):

    results_dict = {}

    specific_rois = [23, 24, 106] #
    # Simulation parameters - define or adjust these according to your needs
    tau, bias, k, s, C, tspan, dt, burn = 1.0, -3, 0.5, 0, 0.1, 1000, 0.01, 100
    T_BOLD, dt_BOLD = tspan - burn, 0.01  # Adjust total time and timestep for the BOLD simulation

    for run in range(10):
        print(f"Running simulation {run + 1}/10")
        # Perform the simulation...
        num_simulations = 100  # Number of simulations to average over

        # Initialize storage for BOLD signals across simulations
        
        s_inputs = None

        # Initialize storage for processed BOLD signals
        processed_bold_signals = {roi: [] for roi in specific_rois}

        for run in range(num_simulations):
            print(f"Running simulation {run + 1}/{num_simulations}")
            x, time, s_input = wilsoncowan_RK2(tau, bias, W, k, s, C, tspan, dt, burn)
            bold_signal = bw(x, T_BOLD, dt_BOLD)

            # Process each ROI's BOLD signal individually
            for roi in specific_rois:
                bold_signal_detrended = detrend(bold_signal[roi])
                print(f"ROI {roi}: Length of detrended BOLD signal = {len(bold_signal_detrended)}")

                filtered_BOLD = apply_bandpass(bold_signal_detrended, 0.02, 0.1, 1, 5)
                discretized_BOLD = discretize_signal(filtered_BOLD.reshape(1, -1), dt_BOLD, 1.0).flatten()
                
                if run == 0:
                    processed_bold_signals[roi] = np.zeros((num_simulations, len(discretized_BOLD)))
                
                processed_bold_signals[roi][run, :] = discretized_BOLD
            # Collect s_input across simulations
            if s_inputs is None:
                s_inputs = np.zeros((num_simulations, s_input.shape[0], s_input.shape[1]))
            s_inputs[run, :, :] = s_input

        # Calculate the average BOLD signal for each ROI
        averaged_bold_signals = {roi: np.mean(processed_bold_signals[roi], axis=0) for roi in specific_rois}
        averaged_s_input = np.mean(s_inputs, axis=0)
        print(f'Shape of averaged_s_input: {averaged_s_input.shape}')



        # Prepare the external input...
        trimmed_s_input = averaged_s_input[:,int(burn / dt):]
        discretized_s_input = discretize_signal(trimmed_s_input.mean(axis=0, keepdims=True), dt, 1.0)
        discretized_s_input=discretized_s_input.flatten()
        mbb_s = [markov_block_bootstrap(discretized_s_input, 1) for run in range(1000)]
        phi_s = [philipp_shuffle(discretized_s_input, 1) for run in range(1000)]
        ran_s = [np.random.permutation(discretized_s_input) for run in range(1000)]

        # Initialize arrays to store shuffled time series for each ROI
        mbb_bold_shuffled = []
        phi_bold_shuffled = []
        ran_bold_shuffled = []



        # For each ROI, calculate TE for ACW time series and shuffled ACW time series, then calculate p-values
        for i, roi in enumerate(specific_rois):
            # Detrend and discretize the BOLD signal for the current ROI
            
            # Generate 100 shuffled time series using each method for the current ROI
            mbb_bold = [markov_block_bootstrap(averaged_bold_signals[roi], 1) for _ in range(1000)]
            phi_bold = [philipp_shuffle(averaged_bold_signals[roi], 1) for _ in range(1000)]
            ran_bold = [np.random.permutation(averaged_bold_signals[roi]) for _ in range(1000)]

            mbb_bold_shuffled.append(mbb_bold)
            phi_bold_shuffled.append(phi_bold)
            ran_bold_shuffled.append(ran_bold)


            # Calculate ACW time series for the shuffled BOLD and s_input signals
            shuffled_acw_bold_mbb = [get_acw_time_series(shuffle, 150, 1) for shuffle in mbb_bold]
            shuffled_acw_bold_phi = [get_acw_time_series(shuffle, 150, 1) for shuffle in phi_bold]
            shuffled_acw_bold_ran = [get_acw_time_series(shuffle, 150, 1) for shuffle in ran_bold]

            shuffled_acw_s_mbb = [get_acw_time_series(shuffle, 150, 1) for shuffle in mbb_s]
            shuffled_acw_s_phi = [get_acw_time_series(shuffle, 150, 1) for shuffle in phi_s]
            shuffled_acw_s_ran = [get_acw_time_series(shuffle, 150, 1) for shuffle in ran_s]

            # Initialize storage for p-values for forward and reverse TE for each shuffle method
            p_values_forward_true = {'mbb': [], 'phi': [], 'ran': []}
            p_values_reverse_true = {'mbb': [], 'phi': [], 'ran': []}
            p_values_forward_fals = {'mbb': [], 'phi': [], 'ran': []}
            p_values_reverse_fals = {'mbb': [], 'phi': [], 'ran': []}
            
            # Original ACW TE calculations
            acw_bold_signals = {roi: [] for roi in specific_rois}
            acw_bold_signals[roi] = [get_acw_time_series(processed_bold_signals[roi][run, :], 150, 1) for run in range(num_simulations)]
            averaged_acw_bold = np.mean(acw_bold_signals[roi], axis=0)
            print(f'Shape of averaged_acw_bold: {averaged_acw_bold.shape}')

            acw_s_input = get_acw_time_series(discretized_s_input, 150, 1)
            te_results_true = calculate_te(acw_s_input, averaged_acw_bold, m=3)
            te_results_reverse_true = calculate_te(averaged_acw_bold, acw_s_input, m=3)

            # Calculate p-values for forward and reverse TE using shuffled ACW time series
            p_values_forward_true['mbb'].append(p_value(acw_s_input, shuffled_acw_bold_mbb, te_results_true))
            p_values_forward_true['phi'].append(p_value(acw_s_input, shuffled_acw_bold_phi, te_results_true))
            p_values_forward_true['ran'].append(p_value(acw_s_input, shuffled_acw_bold_ran, te_results_true))

            p_values_reverse_true['mbb'].append(p_value(acw_bold_signals[roi], shuffled_acw_s_mbb, te_results_reverse_true))
            p_values_reverse_true['phi'].append(p_value(acw_bold_signals[roi], shuffled_acw_s_phi, te_results_reverse_true))
            p_values_reverse_true['ran'].append(p_value(acw_bold_signals[roi], shuffled_acw_s_ran, te_results_reverse_true))

            # Standard analysis without ACW time series

            # Corrected to use discretized_s_input.flatten() and filtered_BOLD directly
            averaged_processed_signals = {roi: np.mean(processed_bold_signals[roi], axis=0) for roi in specific_rois}
            te_results_fals = calculate_te(discretized_s_input.flatten(), averaged_processed_signals[roi], m=3)

            te_results_reverse_fals = calculate_te(averaged_processed_signals[roi], discretized_s_input.flatten(), m=3)

            # Corrected to pass the entire list of shuffled series

            p_values_forward_fals['mbb'].append(p_value(discretized_s_input.flatten(), mbb_bold, te_results_fals))

            p_values_forward_fals['phi'].append(p_value(discretized_s_input.flatten(), phi_bold, te_results_fals))

            p_values_forward_fals['ran'].append(p_value(discretized_s_input.flatten(), ran_bold, te_results_fals))

            p_values_reverse_fals['mbb'].append(p_value(averaged_processed_signals[roi], mbb_s, te_results_reverse_fals))

            p_values_reverse_fals['phi'].append(p_value(averaged_processed_signals[roi], phi_s, te_results_reverse_fals))

            p_values_reverse_fals['ran'].append(p_value(averaged_processed_signals[roi], ran_s, te_results_reverse_fals))
            # Store results for the current ROI in the results dictionary
            if roi not in results_dict:
                results_dict[roi] = {
                    'te_results_true': [],
                    'te_results_reverse_true': [],
                    'p_values_forward_true': {'mbb': [], 'phi': [], 'ran': []},
                    'p_values_reverse_true': {'mbb': [], 'phi': [], 'ran': []},
                    'te_results_fals': [],
                    'te_results_reverse_fals': [],
                    'p_values_forward_fals': {'mbb': [], 'phi': [], 'ran': []},
                    'p_values_reverse_fals': {'mbb': [], 'phi': [], 'ran': []}
                }

            # Append the current run's TE results and p-values to the ROI's entry in results_dict
            results_dict[roi]['te_results_true'].append(te_results_true)
            results_dict[roi]['te_results_reverse_true'].append(te_results_reverse_true)
            results_dict[roi]['te_results_fals'].append(te_results_fals)
            results_dict[roi]['te_results_reverse_fals'].append(te_results_reverse_fals)

            # Since p_values_forward and p_values_reverse contain lists of p-values for the current run, we append them directly
            for method in ['mbb', 'phi', 'ran']:
                results_dict[roi]['p_values_forward_true'][method].append(p_values_forward_true[method][
                                                                         -1])  # Assuming p_values_forward[method] is a list of p-values for the current run
                results_dict[roi]['p_values_reverse_true'][method].append(p_values_reverse_true[method][
                                                                         -1])  # Assuming p_values_reverse[method] is a list of p-values for the current run
                results_dict[roi]['p_values_forward_fals'][method].append(p_values_forward_fals[method][
                                                                         -1])  # Assuming p_values_forward[method] is a list of p-values for the current run
                results_dict[roi]['p_values_reverse_fals'][method].append(p_values_reverse_fals[method][
                                                                         -1])  # Assuming p_values_reverse[method] is a list of p-values for the current run
    return results_dict
def export_results_to_csv(results_dict, csv_path):
    csv_rows = []
    for roi, data in results_dict.items():
        num_runs = len(data['te_results_true'])
        for run_index in range(num_runs):
            for method in ['mbb', 'phi', 'ran']:
                for tau_index in range(len(data['te_results_true'][run_index])):
                    row = {
                        'ROI': roi,
                        'Run': run_index + 1,
                        'Tau': tau_index + 1,
                        'TE Value True': data['te_results_true'][run_index][tau_index],
                        'TE Reverse Value True': data['te_results_reverse_true'][run_index][tau_index],
                        'P-Value Forward True MBB': data['p_values_forward_true']['mbb'][run_index][tau_index],
                        'P-Value Reverse True MBB': data['p_values_reverse_true']['mbb'][run_index][tau_index],
                        'TE Value False': data['te_results_fals'][run_index][tau_index],
                        'TE Reverse Value False': data['te_results_reverse_fals'][run_index][tau_index],
                        'P-Value Forward False MBB': data['p_values_forward_fals']['mbb'][run_index][tau_index],
                        'P-Value Reverse False MBB': data['p_values_reverse_fals']['mbb'][run_index][tau_index],
                    }
                    # Add phi and ran values
                    row.update({
                        'P-Value Forward True PHI': data['p_values_forward_true']['phi'][run_index][tau_index],
                        'P-Value Reverse True PHI': data['p_values_reverse_true']['phi'][run_index][tau_index],
                        'P-Value Forward True RAN': data['p_values_forward_true']['ran'][run_index][tau_index],
                        'P-Value Reverse True RAN': data['p_values_reverse_true']['ran'][run_index][tau_index],
                        'P-Value Forward False PHI': data['p_values_forward_fals']['phi'][run_index][tau_index],
                        'P-Value Reverse False PHI': data['p_values_reverse_fals']['phi'][run_index][tau_index],
                        'P-Value Forward False RAN': data['p_values_forward_fals']['ran'][run_index][tau_index],
                        'P-Value Reverse False RAN': data['p_values_reverse_fals']['ran'][run_index][tau_index],
                    })
                    csv_rows.append(row)

    # Convert the list of rows to a DataFrame and then to a CSV file
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    print(f'Results saved to {csv_path}.')



# Example usage
ACW = False  # Set to False to use raw time series instead
results_dict = run_simulation_and_analysis(ACW=ACW)  # Your results_dict from the function
csv_path = '/Users/satrokommos/Documents/9thsem/code/data/Text/wit-rec15-simulation.csv'
export_results_to_csv(results_dict, csv_path)









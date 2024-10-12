from scipy.signal import butter, lfilter
import numpy as np
import scipy.io as sio
import scipy.signal as signal

from sklearn.cross_decomposition import CCA


def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

# Path: core\utils.py
def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

# Path: core\utils.py
def apply_filter_to_segments(segments, lowcut, highcut, fs, order=6):
    N = segments.shape[0]
    dim = segments.shape[2]
    filtered_segments = np.empty((N, 1500, dim))
    for i in range(N):
        segment = segments[i]
        filtered_segments[i] = butter_bandpass_filter(segment, lowcut, highcut, fs, order=order)
    return filtered_segments

# Path: core\utils.py
def apply_filter_to_signal(signal, lowcut, highcut, fs, order=6):
    dim = signal.shape[1]
    filtered_signal = np.empty((1500, dim))
    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order=order)
    return filtered_signal

# Path: core\utils.py
def apply_filter_to_df(df, lowcut, highcut, fs, order=6):
    dim = df.shape[1]
    filtered_df = np.empty((df.shape[0], dim))
    filtered_df = butter_bandpass_filter(df, lowcut, highcut, fs, order=order)
    return filtered_df

# Path: core\utils.py
def apply_filter_to_df_with_labels(df, lowcut, highcut, fs, order=6):
    dim = df.shape[1]
    filtered_df = np.empty((df.shape[0], dim))
    filtered_df = butter_bandpass_filter(df, lowcut, highcut, fs, order=order)
    filtered_df['labels'] = df['labels']
    return filtered_df


# ****************************** #
# *********** FBCCA ************ #
# ****************************** #

# Function 5: generate_ref_signal
# Input: Freq_phase_path, SSVEP frecuencies (targets), number of samples, number of harmonics, sampling frequency
# Output: reference signal
def generate_ref_signal(Freq_phase_path: str, freqs: list, N: int, n_harmonics: int, fs: int) -> np.ndarray:
    data = sio.loadmat(Freq_phase_path)
    index_freqs = data['freqs'].round(1).reshape(-1).tolist()
    phases = data['phases'].reshape(-1)  # phases

    # Reference signal
    ref_signal = np.zeros((N, len(freqs), n_harmonics * 2,))
    # For each frequency
    f = 0
    for frequency in freqs:
        phase = phases[index_freqs.index(round(frequency, 1))]
        # phase = 3
        
        # For each harmonic
        for i in range(0, n_harmonics * 2, 2):
            # Sinusoid
            ref_signal[:, f, i] = np.sin(2 * np.pi * frequency * (i/2 + 1) * np.arange(N) / fs + phase)
            # ref_signal[:, f, i] = np.arange(N)
            # Cosinusoid
            ref_signal[:, f, i + 1] = np.cos(2 * np.pi * frequency * (i/2 + 1) * np.arange(N) / fs + phase)
       
        f+=1

    return ref_signal.swapaxes(0, 2)

# Function 1: filter_bank_analysis
# Input: signal, sampling frequency, number of sub-bands, filter bank design
# Output: sub-band signals
def filter_bank_analysis(in_signal, fs, n_subbands, filter_bank_design, low_freq, up_freq):
    # Number of channels
    n_channels = in_signal.shape[0]
    # Number of samples
    n_samples = in_signal.shape[1]

    gpass, gstop, Rp = 3, 40, 0.5
    highcut_pass, highcut_stop = 80, 90

    # Filter bank design
    if filter_bank_design == 'M1': # N_subbands of the same length
        # Sub-band signals
        subband_signals = np.zeros((n_subbands, n_channels, n_samples))
        # For each sub-band
        for i in range(0, n_subbands):
            
            passband = low_freq * (i + 2)
            stopband = low_freq * (i + 1)
            
            Wp = [passband / (fs/2), highcut_pass / (fs/2)]
            Ws = [stopband / (fs/2), highcut_stop / (fs/2)]
            
            N, Wn = signal.cheb1ord(Wp, Ws, gpass, gstop)
            
            # Sub-band filter
            b, a = signal.cheby1(N, 0.1, Wn, 'bandpass')
            # Filtered signal
            subband_signals[i, :, :] = signal.filtfilt(b, a, in_signal)
    
    elif filter_bank_design == 'M2':
        raise NotImplementedError()
    
    elif filter_bank_design == 'M3':
        raise NotImplementedError()
    
    else:
        raise NotImplementedError("Design not supported")

    return subband_signals

#  - Define a function to perform CCA between two multi-dimensional variables. Use the CCA module from sklearn.cross_decomposition.
# Function 2: cca
# Input: signal 1, signal 2
# Output: correlation vector
def cca(signal1, signal2):
    # Canonical correlation analysis
    cca = CCA(1)
    
    # Fit the model with signal 1 and signal 2
    cca.fit(signal1.T, signal2.T)

    # Correlation vector
    X_c, Y_c = cca.transform(signal1.T, signal2.T)

    return X_c, Y_c


def feature_extraction(subband_signals, ref_signals, n_freq):
    """
    Extracts features from sub-band signals and reference signals using Canonical Correlation Analysis (CCA).
    Parameters:
        subband_signals (numpy.ndarray): A 3D array of shape (n_subbands, n_samples, n_timepoints) representing the sub-band signals.
        ref_signals (numpy.ndarray): A 3D array of shape (n_samples, n_freq, n_timepoints) representing the reference signals.
        n_freq (int): The number of frequency components in the reference signals.
    Returns:
        int: The predicted class based on the extracted features.
    """
    # Number of sub-band signals
    n_subbands = subband_signals.shape[0]
    # Feature vector
    feature_vector = np.zeros(ref_signals.shape[1])
    # Class predicted
    fb_coefs = [pow(i, -1.25) + 0.25 for i in range(1, n_subbands + 1)]
    result = 0
    # For each sub-band signal
    for sub in range(n_subbands):
        for freq in range(ref_signals.shape[1]):
            # Correlation vector
            X_c, Y_c = cca(subband_signals[sub, :, :], ref_signals[:, freq, :])
            # Pearson correlation coefficient
            correl = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
        
            feature_vector[freq] = np.max(correl)
            result += (fb_coefs[sub] * (feature_vector ** 2))
        
        predicted = np.argmax(result) + 1

    return predicted


"""
    Perform Frequency-Based Common Spatial Pattern Analysis (FBCSP) on the input signal.

    Args:
        in_signal (ndarray): Input signal.
        fs (int): Sampling frequency of the input signal.
        n_subbands (int): Number of subbands to divide the signal into.
        filter_bank_design (str): Design of the filter bank.
        w (int): Window size for feature extraction.
        ref_signals (ndarray): Reference signals for feature extraction.
        lowest_freq (float): Lowest frequency of the subbands.
        upmost_freq (float): Highest frequency of the subbands.

    Returns:
        ndarray: Predicted target frequency.

"""
def fbcca(in_signal, fs, n_subbands, filter_bank_design, w, ref_signals, lowest_freq, upmost_freq):
    
    # Sub-band signals
    in_signal = in_signal.swapaxes(0, 1)
    subband_signals = filter_bank_analysis(in_signal, fs, n_subbands, filter_bank_design, lowest_freq, upmost_freq)
    # Target frequency
    predicted = feature_extraction(subband_signals, ref_signals, w)
    return predicted

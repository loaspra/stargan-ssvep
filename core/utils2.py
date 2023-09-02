# Now, create a bandpass filter to filter the signal between 2 and 54 Hz
# Use a butterworth filter of order 6
from scipy.signal import butter, lfilter

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
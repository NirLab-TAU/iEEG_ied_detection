import mne
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import stats, signal
import pandas as pd
from pathlib import Path, PurePath


# Thresholds for standard deviation (z-score) to detect spikes
thresh_amp = 5
thresh_grad = 5
thresh_env = 5

# Block size for signal processing, in seconds
block_size_sec = 30

# Filter settings: low_pass and high_pass frequencies for band-pass filtering
low_pass = None
high_pass = 250

# Maximum allowed spike length in milliseconds (used to limit spike duration)
max_spike_length_ms = 70

# List to store detected spikes information (e.g., indexes, amplitudes, channels)
spikes_list = []


def get_markers(data, index_above_threshold, thresh_type, sr, channel_name):
    max_markers_index, max_marker_value = [], []

    # Define the maximum number of points considered as a single spike based on the sampling rate (sr)
    max_spike_points = sr / (1000 / max_spike_length_ms)
    counter = 1
    curr_spike = [thresh_type, index_above_threshold[0]]

    for j in range(len(index_above_threshold)):
        # Check if the next index is contiguous (i.e., part of the same spike)
        if j + 1 < len(index_above_threshold) and index_above_threshold[j + 1] - index_above_threshold[j] == 1:
            counter += 1
        # If the spike has ended
        else:
            # Check if the spike duration is within the allowed maximum length
            if counter <= max_spike_points:
                # Find the maximum and minimum values within the current spike window
                max_value = data[index_above_threshold[j - counter + 1]: index_above_threshold[j] + 1].max()
                min_value = data[index_above_threshold[j - counter + 1]: index_above_threshold[j] + 1].min()
                # Choose the peak value (either maximum or minimum) based on absolute value
                value = max_value if abs(max_value) > abs(min_value) else min_value
                # Find the exact index of the peak value within the current spike window
                index = np.intersect1d(np.where(data == value)[0], index_above_threshold[j - counter + 1: j + 1])[0]
                # Add the current spike to the spikes list
                max_marker_value.append(value)
                max_markers_index.append(index)
                curr_spike.extend((index_above_threshold[j], index, value, channel_name))
                spikes_list.append(curr_spike)

            # Prepare for the next spike if there are more indices to process
            if j + 1 < len(index_above_threshold):
                curr_spike = [thresh_type, index_above_threshold[j + 1]]
                counter = 1

    return np.array(max_markers_index), np.array(max_marker_value)


def detect(data, channel, sampling_rate, plot=True):
    # Calculate the number of data points in each block
    points_in_block = block_size_sec * sampling_rate
    number_of_blocks = math.floor(len(data) / points_in_block)
    if plot:
        plt.plot(data, alpha=0.8)

    # Iterate over the signal blocks (each block is 30 seconds)
    for i in range(number_of_blocks):
        curr_block = data[i * points_in_block: (i + 1) * points_in_block]

        # Check amplitude threshold
        z_score_amp = stats.zscore(curr_block)
        points_above_thresh_amp = z_score_amp[z_score_amp > thresh_amp]

        # Check gradient threshold
        gradient_diff = np.diff(curr_block)
        z_score_grad = stats.zscore(np.insert(gradient_diff, 0, 0))
        points_above_thresh_grad = z_score_grad[z_score_grad > thresh_grad]

        # Get the common indices that exceed both amplitude and gradient thresholds
        if len(points_above_thresh_amp) > 0 and len(points_above_thresh_grad) > 0:
            index_above_thresh_amp_grad = np.intersect1d(points_above_thresh_amp, points_above_thresh_grad)
            if len(index_above_thresh_amp_grad) > 0:
                # Find and save the specific indices of the maximum values
                max_markers_index_amp_grad, max_marker_value_amp_grad = get_markers(data,
                                                                                    index_above_thresh_amp_grad,
                                                                                    'amp_grad', sampling_rate, channel)
                if plot:
                    plt.scatter(max_markers_index_amp_grad, data[max_markers_index_amp_grad] if len(
                        max_markers_index_amp_grad) > 0 else [], marker='s', color='green')

        # Check envelope threshold
        filtered_block = mne.filter.filter_data(curr_block, sampling_rate, low_pass, high_pass)
        env_block = abs(signal.hilbert(filtered_block))
        z_score_env = stats.zscore(env_block)
        points_above_thresh_env = z_score_env[z_score_env > thresh_env]

        # If points above the envelope threshold exist, process them
        if len(points_above_thresh_env) > 0:
            index_above_threshold_env = (z_score_env > thresh_env).nonzero()[0] + i * points_in_block
            max_markers_index_env, max_marker_value_env = get_markers(data, index_above_threshold_env, 'env',
                                                                      sampling_rate, channel)
            if plot:
                plt.scatter(max_markers_index_env,
                            data[max_markers_index_env] if len(max_markers_index_env) > 0 else [], marker='o',
                            color='blue', s=15)

    if plot:
        plt.close()


# Main loop to run the detection for all subjects and channels
for subj in range(1, 26):
    # Define the path to the subject's data
    dir = Path(fr'.\ieeg_ieds_bids\sub-{subj}\ieeg\\')
    fname = PurePath(dir, f"sub-{subj}_task-sleep_ieeg.edf")

    # Load the raw EEG data for the subject
    raw = mne.io.read_raw_edf(fname)
    sampling_rate = int(raw.info['sfreq'])

    # Iterate over all channels (beside bipolar channels) and run detection
    for chan in [x for x in raw.ch_names if '-' not in x]:
        chan_data = raw.copy().pick([chan]).get_data()[0]
        detect(chan_data, chan, sampling_rate, False)
        print(f'finish channel {chan}')

    # Create a DataFrame from the detected spikes list and save it to a CSV file
    spikes_df = pd.DataFrame(spikes_list,
                             columns=['threshold_type', 'first_index', 'last_index', 'max_index', 'max_amp', 'channel'])

    spikes_df.to_csv(PurePath(fr'C:\repos\depth_ieds\thresh_5\\', f'P{subj}_spikes.csv'))
    # Clear the spikes list for the next subject
    spikes_list = []

    print(f'finish subj {subj}')

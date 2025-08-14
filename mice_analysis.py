import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.decomposition import PCA
import pickle
from os.path import join
from collections import defaultdict
import scipy
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
from scipy.signal import butter, filtfilt, welch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.linalg import orthogonal_procrustes
from decodanda import Decodanda

def replace_key_recursive(d, old_key, new_key):
			if isinstance(d, dict):
					keys = list(d.keys())
					for key in keys:
						if key == old_key:
							d[new_key] = d.pop(old_key)
						# Recursively check the value
						replace_key_recursive(d.get(key, {}), old_key, new_key)
			elif isinstance(d, list):
				for item in d:
					replace_key_recursive(item, old_key, new_key)

def rename_dict_keys(d, seq_types, mouse_ids, sessions_by_id, key_to_pat_dict):
	# rename keys in the data dictionary to match the key_to_pat_dict
	for seq_type in seq_types:
		old_key = seq_type
		new_key = key_to_pat_dict[seq_type]
		replace_key_recursive(data, old_key, new_key)
		
	return d


def make_data_seqtype(
    seq_types,
    mouse_ids,
    sessions_by_id,
    data,
    which_zscore='across_time',
    shuffle_seqtypes=False,
    pseudopop=True
):
    """
    Create pseudopopulation or session-wise z-scored neural data for sequence types.

    Args:
        seq_types (list): List of sequence types (e.g. ['A-0', 'A-1']).
        mouse_ids (list): Mouse IDs to include.
        sessions_by_id (dict): Dict mapping mouse_id -> list of sessions.
        data (dict): data[mouse_session][seq_type] = ndarray (trials, neurons, timepoints).
        which_zscore (str): 'across_time', 'across_trials', or 'across_trials_and_time'.
        shuffle_seqtypes (bool): If True, shuffle sequence identities within each session.
        pseudopop (bool): If True, returns pseudopopulation averaged across trials. Otherwise returns full trial-wise data.

    Returns:
        If pseudopop=True:
            ndarray of shape (num_seqtypes, timepoints, total_neurons)
        If pseudopop=False:
            dict of form data_new[mouse_session][seq_type] = array of shape (trials, timepoints, neurons)
    """
    assert which_zscore in ['across_time', 'across_trials', 'across_trials_and_time'], \
        f"Invalid z-score method: {which_zscore}"

    data_by_seq = defaultdict(list)
    data_new = {}

    for mouse_id in mouse_ids:
        for session in sessions_by_id[mouse_id]:
            key = f'{mouse_id}_{session}'

            if key not in data:
                continue

            data_new[key] = {}

            available_seqtypes = [st for st in seq_types if st in data[key]]
            if shuffle_seqtypes:
                shuffled_seqtypes = np.random.permutation(available_seqtypes)
                mapping = dict(zip(available_seqtypes, shuffled_seqtypes))
            else:
                mapping = {st: st for st in available_seqtypes}

            for orig_seqtype in available_seqtypes:
                new_seqtype = mapping[orig_seqtype]
                data_single_sess = data[key][orig_seqtype]  # (trials, neurons, timepoints)

                # Z-score
                if which_zscore == 'across_trials_and_time':
                    temp = data_single_sess.transpose(0, 2, 1).reshape(-1, data_single_sess.shape[1])
                    temp = zscore(temp, axis=0)
                    data_single_sess = temp.reshape(
                        data_single_sess.shape[0], data_single_sess.shape[2], data_single_sess.shape[1]
                    ).transpose(0, 2, 1)
                elif which_zscore == 'across_trials':
                    data_single_sess = zscore(data_single_sess, axis=0)
                elif which_zscore == 'across_time':
                    data_single_sess = zscore(data_single_sess, axis=2)

                data_single_sess[np.isnan(data_single_sess)] = 0.0

                if pseudopop:
                    avg = np.nanmean(data_single_sess, axis=0)  # (neurons, timepoints)
                    data_by_seq[new_seqtype].append(avg)
                else:
                    # Format like (trials, timepoints, neurons) for consistency with downstream use
                    data_new[key][new_seqtype] = data_single_sess.transpose(0, 2, 1)

    if pseudopop:
        # Stack pseudopopulation
        data_pseudopop = []
        for seq_type in seq_types:
            arrays = data_by_seq[seq_type]
            if not arrays:
                raise ValueError(f"No data for sequence type '{seq_type}'")

            time_lengths = [arr.shape[1] for arr in arrays]
            if len(set(time_lengths)) != 1:
                raise ValueError(f"Inconsistent number of timepoints for '{seq_type}': {time_lengths}")

            stacked = np.concatenate(arrays, axis=0)  # (neurons, timepoints)
            stacked = stacked.T  # (timepoints, total_neurons)
            data_pseudopop.append(stacked)

        data_pseudopop = np.stack(data_pseudopop, axis=0)  # (seq_types, timepoints, total_neurons)
        return data_pseudopop
    else:
        return data_new

def make_data_freqs(frequencies, mouse_ids, sessions_by_id, data, pseudopop=False, shuffle_freqs=False):

	freq_slices = {
		'A0': {'ABCD0': [(25, 40)], 'ABBA0': [(25, 40), (100, 115)]},
		'B0': {'ABCD0': [(50, 65)], 'ABBA0': [(50, 65), (75, 90)]},
		'C0': {'ABCD0': [(75, 90)], 'ABBA0': []},
		'D0': {'ABCD0': [(100, 115)], 'ABBA0': []},
		'A1': {'ABCD1': [(25, 40)], 'ABBA1': [(25, 40), (100, 115)]},
		'B1': {'ABCD1': [(50, 65)], 'ABBA1': [(50, 65), (75, 90)]},
		'C1': {'ABCD1': [(75, 90)], 'ABBA1': []},
		'D1': {'ABCD1': [(100, 115)], 'ABBA1': []}
	}

	which_zscore = 'across_time'
	data_new = {}

	# Optional: shuffle frequency labels within each session
	if shuffle_freqs:
		from copy import deepcopy
		import random
		shuffled_frequencies_by_session = {}
		for mouse_id in mouse_ids:
			for session in sessions_by_id[mouse_id]:
				shuffled = deepcopy(frequencies)
				random.shuffle(shuffled)
				shuffled_frequencies_by_session[(mouse_id, session)] = dict(zip(frequencies, shuffled))
	else:
		shuffled_frequencies_by_session = {
			(mouse_id, session): {f: f for f in frequencies}
			for mouse_id in mouse_ids for session in sessions_by_id[mouse_id]
		}

	for mouse_id in mouse_ids:
		for session in sessions_by_id[mouse_id]:
			key = f'{mouse_id}_{session}'
			data_new[key] = {freq: [] for freq in frequencies}
			for freq in frequencies:
				shuffled_freq = shuffled_frequencies_by_session[(mouse_id, session)][freq]
				for seq_type in freq_slices[shuffled_freq]:
					data_single_sess = data[key][seq_type]
					for slc in freq_slices[shuffled_freq][seq_type]:
						if len(data_new[key][freq]) == 0:
							data_new[key][freq] = data_single_sess[:, :, slc[0]:slc[1]]
						else:
							data_new[key][freq] = np.concatenate(
								(data_new[key][freq], data_single_sess[:, :, slc[0]:slc[1]]), axis=0
							)
			for freq in frequencies:
				data_new[key][freq] = np.array(data_new[key][freq])

	data_pseudopop_temp = []
	for frequency in frequencies:
		for mouse_id in mouse_ids:
			for session in sessions_by_id[mouse_id]:
				key = f'{mouse_id}_{session}'
				data_single_sess = data_new[key][frequency]
				num_trials, num_neurons, num_timepts = data_single_sess.shape
				if which_zscore == 'across_trials_and_time':
					data_single_sess = data_single_sess.transpose(0, 2, 1)
					data_single_sess = data_single_sess.reshape(-1, num_neurons)
					data_single_sess = zscore(data_single_sess, axis=0)
					data_single_sess = data_single_sess.reshape(num_trials, num_timepts, num_neurons)
					data_single_sess = data_single_sess.transpose(0, 2, 1)
				elif which_zscore == 'across_trials':
					data_single_sess = zscore(data_single_sess, axis=0)
				elif which_zscore == 'across_time':
					data_single_sess = zscore(data_single_sess, axis=2)

				data_single_sess[np.isnan(data_single_sess)] = 0.0

				if pseudopop:
					data_single_sess_avg_across_trials = np.nanmean(data_single_sess, axis=0)
					data_pseudopop_temp.append(data_single_sess_avg_across_trials)
				else:
					data_new[key][frequency] = data_single_sess.transpose(0, 2, 1)

	if pseudopop:
		data_pseudopop_stacked = np.vstack(data_pseudopop_temp)
		assert data_pseudopop_stacked.shape[0] % len(frequencies) == 0
		data_pseudopop = data_pseudopop_stacked.reshape(
			len(frequencies),
			data_pseudopop_stacked.shape[0] // len(frequencies),
			data_pseudopop_stacked.shape[1]
		).transpose(0, 2, 1)
		return data_pseudopop
	else:
		return data_new



def plot_PCA(data_embedding, seq_types, frequencies, key_to_pat_dict, n_components, frac_variance, figname, N, which_feature='seqtype'):
	fig, ax = plt.subplots(2,5, figsize=(20, 8))
	ax = ax.flatten()
	colors = ['Blue', 'Purple', 'Green', 'Red']

	for comp in range(n_components):
		for i in range(len(seq_types)):
			if which_feature == 'seqtype':
				ax[comp].plot(np.arange(np.shape(data_embedding)[1]), data_embedding[i,:,comp], color=colors[i], label=key_to_pat_dict[seq_types[i]])
			elif which_feature == 'frequency':
				ax[comp].plot(np.arange(np.shape(data_embedding)[1]), data_embedding[i,:,comp], color=colors[i], label=frequencies[i])
			# ax[comp].axvspan(0+25, 15+25, color='red', lw=0, alpha=0.03)
			# ax[comp].axvspan(25+25, 40+25, color='red', lw=0, alpha=0.03)
			# ax[comp].axvspan(50+25, 65+25, color='red', lw=0, alpha=0.03)
			# ax[comp].axvspan(75+25, 90+25, color='red', lw=0, alpha=0.03)
			# ax[comp].set_xticks(np.arange(0, 126, 20))
			# ax[comp].set_xticklabels(np.arange(0, 1260, 200))
			ax[comp].set_title(f'PC {comp+1} ({frac_variance[comp]:.2f} variance)', fontsize=16)
			# ax[comp].set_ylim(-4, 5)
	ax[5].set_xlabel('Time (ms)')
	ax[5].set_ylabel('Component Value')
	ax[0].legend()
	fig.suptitle(f'{N} neurons across sessions')
	fig.savefig(f'figs/pca_components_{figname}_{which_feature}.svg', bbox_inches='tight')	


def plot_3D_PCA_trajectory(data_embedding, figname, which_feature='seqtype'):
# data_embedding: shape (n_patterns, n_timepoints, n_components)

	colormaps = [cm.Blues, cm.Purples, cm.Greens, cm.Reds]
	pc_triplets = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
	n_triplets = len(pc_triplets)

	fig = plt.figure(figsize=(8 * n_triplets, 6))
	for idx, (pcx, pcy, pcz) in enumerate(pc_triplets):
		print(idx, pcx, pcy, pcz)
		ax = fig.add_subplot(1, n_triplets, idx + 1, projection='3d')
		for patt_idx, Y in enumerate(data_embedding):
			# Check if enough components exist
			if Y.shape[1] <= max(pcx, pcy, pcz):
				continue
			x, y, z = Y[:, pcx], Y[:, pcy], Y[:, pcz]
			points = np.array([x, y, z]).T
			segments = np.array([[points[i], points[i+1]] for i in range(len(points) - 1)])
			norm_time = np.linspace(0.2, 0.8, len(segments))
			colors = colormaps[patt_idx % len(colormaps)](norm_time)
			lc = Line3DCollection(segments, colors=colors, linewidth=2)
			ax.add_collection3d(lc)
		ax.set_xlabel(f'PC{pcx+1}')
		ax.set_ylabel(f'PC{pcy+1}')
		ax.set_zlabel(f'PC{pcz+1}')
		ax.set_title(f'PC{pcx+1} vs PC{pcy+1} vs PC{pcz+1}')
		# Autoscale
		all_xyz = np.vstack([Y[:, [pcx, pcy, pcz]] for Y in data_embedding if Y.shape[1] > max(pcx, pcy, pcz)])
		ax.auto_scale_xyz(all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2])

	fig.tight_layout()
	fig.savefig(f"figs/PCA_3D_triplets_{figname}_{which_feature}.svg", bbox_inches="tight", dpi=600)


def bandstop_filter(data, lowcut=4, highcut=12, fs=100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data)


def plot_bandpass(raw_signals, figname, freq_types, seq_types, n_components, which_feature='seqtype'):
	x = len(seq_types) if which_feature == 'seqtype' else len(freq_types)
	fig, ax = plt.subplots(3, x, figsize=(3*x, 10))

	colors = ['Blue', 'Purple', 'Green', 'Red']
	filtered_signal = np.zeros_like(raw_signals)

	fs = 100 if which_feature == 'seqtype' else 10

	for i, raw_signal in enumerate(raw_signals):
		if which_feature == 'seqtype':
			ax[0,i].axvspan(0+25, 15+25, color='red', lw=0, alpha=0.1)
			ax[0,i].axvspan(25+25, 40+25, color='red', lw=0, alpha=0.1)
			ax[0,i].axvspan(50+25, 65+25, color='red', lw=0, alpha=0.1)
			ax[0,i].axvspan(75+25, 90+25, color='red', lw=0, alpha=0.1)
			ax[1,i].axvspan(0+25, 15+25, color='red', lw=0, alpha=0.1)
			ax[1,i].axvspan(25+25, 40+25, color='red', lw=0, alpha=0.1)
			ax[1,i].axvspan(50+25, 65+25, color='red', lw=0, alpha=0.1)
			ax[1,i].axvspan(75+25, 90+25, color='red', lw=0, alpha=0.1)

		for j in range(n_components):
			signal = raw_signal[:, j]
			# Function to apply bandpass filter and plot the results
			# raw_signal: your ephys trace

			ax[0,i].plot(np.arange(len(signal)), signal, label='Raw Signal', color='gray', alpha=0.2)

			# Plot filtered signal
			if which_feature == 'seqtype':
				filtered = bandstop_filter(signal, fs=fs)
				filtered_signal[i, :, j] = filtered
				ax[1,i].plot(np.arange(len(signal)), filtered, label='Filtered Signal (Theta Band)', ls='--', color='gray', alpha=0.2)

			# PSD (Power Spectral Density)
			frequencies, power = welch(signal, nperseg=64, fs=fs)
			ax[2,i].semilogy(frequencies, power, color='gray', alpha=0.1)

		if which_feature == 'seqtype':
			ax[0,i].set_title(f'Seq Type: {seq_types[i]}', fontsize=16)
			ax[1,i].set_title(f'Seq Type: {seq_types[i]}', fontsize=16)
		else:
			ax[0,i].set_title(f'Freq Type: {freq_types[i]}', fontsize=16)
			ax[1,i].set_title(f'Freq Type: {freq_types[i]}', fontsize=16)

		ax[0,i].set_xlabel('Time (s)')
		ax[0,i].set_ylabel('Amplitude')
		ax[0,i].set_xticks(np.arange(0, len(signal), 30))
		ax[0,i].set_xticklabels(np.arange(0, len(signal)*10, 30*10))
		
		ax[1,i].set_xlabel('Time (s)')
		ax[1,i].set_ylabel('Amplitude')
		ax[1,i].set_xticks(np.arange(0, len(signal), 30))
		ax[1,i].set_xticklabels(np.arange(0, len(signal)*10, 30*10))

		ax[2,i].set_xlabel('Frequency (Hz)')
		ax[2,i].set_ylabel('Power')
		ax[2,i].axvspan(4, 12, color='gray', alpha=0.3)

		ax[0,i].plot(np.arange(len(signal)), np.mean(raw_signal, axis=1), label='Raw Signal', color='black')
		ax_inset = inset_axes(ax[0,i], width="40%", height="30%", loc=1)  # loc=1 is upper right
		ax_inset.plot(np.arange(len(signal)), np.mean(raw_signal, axis=1), label='Raw Signal', color='black')
		ax_inset.set_xticks(np.arange(0, len(signal), 30))
		ax_inset.set_xticklabels(np.arange(0, len(signal)*10, 30*10))

		
		ax[1,i].plot(np.arange(len(signal)), np.mean(filtered_signal[i], axis=1), label='Filtered Signal', color='black')
		ax_inset = inset_axes(ax[1,i], width="40%", height="30%", loc=1)  # loc=1 is upper right
		ax_inset.plot(np.arange(len(signal)), np.mean(filtered_signal[i], axis=1), label='Raw Signal', color='black')
		ax_inset.set_xticks(np.arange(0, len(signal), 30))
		ax_inset.set_xticklabels(np.arange(0, len(signal)*10, 30*10))

	fig.tight_layout()
	fig.savefig(f"figs/filter_{figname}_{which_feature}.svg", bbox_inches="tight", dpi=600)
	
	return filtered_signal


def plot_firing_rate(mouse_ids, fignames, frequencies, n_components, which_feature, key_to_pat_dict, sessions_by_id, data):
	
	for mouse_id, figname in zip(mouse_ids, fignames):
		print(f"Processing mouse: {mouse_id}")

		if which_feature == 'seqtype':
			data_pseudopop = make_data_seqtype(
				[key_to_pat_dict[st] for st in seq_types],
				mouse_id,
				sessions_by_id,
				data,
				pseudopop=True
			)

		elif which_feature == 'frequency':
			data_pseudopop = make_data_freqs(
				frequencies,
				mouse_id,
				sessions_by_id,
				data,
				pseudopop=True
			)
		else:
			raise ValueError("Invalid feature type. Choose 'seqtype' or 'frequency'.")
	
		plot_bandpass(data_pseudopop, figname, frequencies, [key_to_pat_dict[st] for st in seq_types], n_components=n_components, which_feature=which_feature)
		

def run_PCA_across_time(
    mouse_ids,
    fignames,
    which_feature,
    seq_types,
    frequencies,
    key_to_pat_dict,
    sessions_by_id,
    data,
    n_components=3
):
    """
    Run PCA across time for a set of mice and plot the results.

    Args:
        mouse_ids (list of str): List of mouse IDs.
        fignames (list of str): Corresponding figure names for saving.
        which_feature (str): Feature type, either 'seqtype' or 'frequency'.
        seq_types (list): List of sequence types (for 'seqtype' feature).
        frequencies (list): List of frequencies (for 'frequency' feature).
        key_to_pat_dict (dict): Dictionary mapping sequence types to patterns.
        sessions_by_id (dict): Mapping from mouse ID to session data.
        data (dict): Neural data.
        n_components (int): Number of PCA components to retain.
    """

    for mouse_id, figname in zip(mouse_ids, fignames):
        print(f"Processing mouse: {mouse_id}")

        if which_feature == 'seqtype':
            data_pseudopop = make_data_seqtype(
                [key_to_pat_dict[st] for st in seq_types],
                mouse_id,
                sessions_by_id,
                data,
                pseudopop=True
            )
        elif which_feature == 'frequency':
            data_pseudopop = make_data_freqs(
                frequencies,
                mouse_id,
                sessions_by_id,
                data,
                pseudopop=True
            )
        else:
            raise ValueError("Invalid feature type. Choose 'seqtype' or 'frequency'.")

        _p, _t, _N = data_pseudopop.shape
        print(f'number of seqtypes = {_p}')
        print(f'number of timepoints = {_t}')
        print(f'number of neurons = {_N}')

        flattened_data = data_pseudopop.reshape(-1, _N)
        print("Flattened data shape:", flattened_data.shape)

        pca = PCA(n_components=n_components)
        _ = pca.fit_transform(flattened_data)
        data_embedding = pca.transform(flattened_data)

        proj_variance = np.cumsum(np.var(data_embedding, axis=0))
        total_variance = np.sum(np.var(flattened_data, axis=0))
        frac_variance = proj_variance / total_variance

        print("proj_variance shape =", proj_variance.shape)
        print("total_variance =", total_variance)
        print("frac_variance =", frac_variance)

        data_embedding = data_embedding.reshape(_p, _t, n_components)

        plot_PCA(data_embedding, seq_types, frequencies, key_to_pat_dict, n_components, frac_variance, figname, _N, which_feature=which_feature)

        plot_3D_PCA_trajectory(data_embedding, figname, which_feature=which_feature)

def plot_PCA_by_session(
    unique_mouse_ids,
    fignames,
    frequencies,
    sessions_by_id,
    data,
    n_components=2,
    output_prefix='pca_freq_scatter'):
    """
    Perform PCA on mean frequency responses for each session of each mouse,
    and plot the first two components in a grid layout.

    Args:
        unique_mouse_ids (list of str): List of unique mouse IDs.
        fignames (list of str): List of figure names for saving.
        frequencies (list): List of stimulus frequencies.
        sessions_by_id (dict): Dictionary mapping each mouse ID to its sessions.
        data (dict): Neural data.
        n_components (int): Number of PCA components (default=2 for 2D plots).
        output_prefix (str): Prefix for output filenames (default='pca_freq_scatter').
    """

    data_by_freq = make_data_freqs(
        frequencies,
        unique_mouse_ids,
        sessions_by_id,
        data,
        pseudopop=False
    )

    for mouse_id, figname in zip(unique_mouse_ids, fignames):
        fig, ax = plt.subplots(4, 5, figsize=(20, 10))  # up to 20 sessions
        ax = ax.flatten()

        for idx_sess, session in enumerate(sessions_by_id[mouse_id]):
            key = f'{mouse_id}_{session}'
            data_session = data_by_freq[key]

            data_freq_mean = []
            freq_sizes = []

            for freq_idx, freq in enumerate(data_session.keys()):
                data_freq = data_session[freq]  # shape: (trials, time, neurons)
                freq_mean = np.mean(data_freq, axis=1)  # mean over time
                data_freq_mean.append(freq_mean)
                freq_sizes.append(freq_mean.shape[0])

            # Stack all frequencies and compute PCA
            data_freq_mean = np.vstack(data_freq_mean)  # shape: (total_trials, neurons)
            freq_sizes = np.append(0, np.cumsum(freq_sizes))

            pca = PCA(n_components=n_components)
            _ = pca.fit_transform(data_freq_mean)
            data_embedding = pca.transform(data_freq_mean)

            if idx_sess < 20:  # plotting only up to 20 sessions
                colors = ['Blue', 'Purple', 'Green', 'Red', 'Orange', 'Pink', 'Brown', 'Gray']
                for j in range(len(freq_sizes) - 1):
                    f_i = freq_sizes[j]
                    f_f = freq_sizes[j + 1]
                    ax[idx_sess].scatter(
                        data_embedding[f_i:f_f, 0],
                        data_embedding[f_i:f_f, 1],
                        color=colors[j],
                        alpha=0.5,
                        label=frequencies[j] if idx_sess == 0 else None
                    )

                ax[idx_sess].set_title(f'Session {session}', fontsize=16)
                ax[idx_sess].set_xlabel('PC1')
                ax[idx_sess].set_ylabel('PC2')

                if idx_sess == 0:
                    ax[idx_sess].legend(loc='upper right', frameon=True, fontsize=12)

        fig.tight_layout()
        fig.savefig(f'figs/{output_prefix}_{figname}.svg', bbox_inches='tight')

def plot_aligned_PCA_components(
	mouse_ids,
	fignames,
	which_feature,
	seq_types,
	frequencies,
	key_to_pat_dict,
	sessions_by_id,
	data,
	output_filename=None,
	num_components=4):

	dim_pairs = [(i, i + 1) for i in range(num_components - 1)]
	fig, ax = plt.subplots(4, len(dim_pairs), figsize=(4 * len(dim_pairs), 12))
	if len(dim_pairs) == 1:
		ax = np.array([[ax[0]], [ax[1]], [ax[2]]])

	# reference_embedding = None
	# reference_embedding_shuffle = None

	for idx, (mouse_id, figname) in enumerate(zip(mouse_ids, fignames)):
		print(f"Processing mouse: {mouse_id}")

		if which_feature == 'seqtype':
			patterns = [key_to_pat_dict[st] for st in seq_types]
			data_pseudopop = make_data_seqtype(patterns, mouse_id, sessions_by_id, data, pseudopop=True, shuffle_seqtypes=False)
			data_pseudopop_shuffle = make_data_seqtype(patterns, mouse_id, sessions_by_id, data, pseudopop=True, shuffle_seqtypes=True)
			labels = patterns
		elif which_feature == 'frequency':
			data_pseudopop = make_data_freqs(frequencies, mouse_id, sessions_by_id, data, pseudopop=True, shuffle_freqs=False)
			data_pseudopop_shuffle = make_data_freqs(frequencies, mouse_id, sessions_by_id, data, pseudopop=True, shuffle_freqs=True)
			labels = frequencies
		else:
			raise ValueError("Invalid feature type. Choose 'seqtype' or 'frequency'.")

		data_pseudopop_mean = np.mean(data_pseudopop, axis=1)
		data_pseudopop_mean_shuffle = np.mean(data_pseudopop_shuffle, axis=1)
		print("Mean data shape:", data_pseudopop_mean.shape)

		# PCA on unshuffled
		pca = PCA(n_components=num_components)
		data_embedding = pca.fit_transform(data_pseudopop_mean)
		frac_variance = pca.explained_variance_ratio_

		# PCA on shuffled
		pca = PCA(n_components=num_components)
		data_embedding_shuffle = pca.fit_transform(data_pseudopop_mean_shuffle)
		frac_variance_shuffle = pca.explained_variance_ratio_

		# Alignment
		if idx == 0:
			reference_embedding = data_embedding
			aligned_embedding = data_embedding
			reference_embedding_shuffle = data_embedding_shuffle
			aligned_embedding_shuffle = data_embedding_shuffle
		else:
			R, x = orthogonal_procrustes(data_embedding, reference_embedding)
			aligned_embedding = data_embedding @ R

			R_shuffle, y = orthogonal_procrustes(data_embedding_shuffle, reference_embedding_shuffle)
			aligned_embedding_shuffle = data_embedding_shuffle @ R_shuffle

		# Plotting
		colors = ['Blue', 'Purple', 'Green', 'Red', 'Orange', 'Pink', 'Brown', 'Gray']
		for i in range(len(aligned_embedding)):
			for plot_idx, (dim_x, dim_y) in enumerate(dim_pairs):
				for row, (embedding, frac_var, title) in enumerate(zip(
					[data_embedding, aligned_embedding, data_embedding_shuffle, aligned_embedding_shuffle],
					[frac_variance, frac_variance, frac_variance_shuffle, frac_variance_shuffle],
					['Original', 'Aligned', 'Shuffled Original', 'Shuffled Aligned'])):

					ax[row, plot_idx].scatter(
						embedding[i, dim_x], embedding[i, dim_y],
						s=4,
						color=colors[i % len(colors)],
						label=labels[i] if figname == 'all' else None
					)
					ax[row, plot_idx].text(
						embedding[i, dim_x], embedding[i, dim_y],
						str(figname),
						fontsize=10,
						color=colors[i % len(colors)]
					)
					ax[row, plot_idx].set_xlabel(f'PC{dim_x + 1} ({frac_var[dim_x]*100:.1f}%)')
					ax[row, plot_idx].set_ylabel(f'PC{dim_y + 1} ({frac_var[dim_y]*100:.1f}%)')
					ax[row, plot_idx].set_title(f'{title}', fontsize=14)

	# Finalize plot
	ax[1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
	fig.tight_layout()

	if output_filename is None:
		output_filename = f'figs/pca_components_aligned_{which_feature}.svg'

	fig.savefig(output_filename, bbox_inches='tight')

	

def between_within_variance_ratio(embedding, labels):
    labels = np.array(labels)
    classes = np.unique(labels)
    overall_mean = np.mean(embedding, axis=0)

    # Between-class variance
    between = 0
    for c in classes:
        class_mean = np.mean(embedding[labels == c], axis=0)
        n_c = np.sum(labels == c)
        between += n_c * np.sum((class_mean - overall_mean) ** 2)

    # Within-class variance
    within = 0
    for c in classes:
        within += np.sum((embedding[labels == c] - np.mean(embedding[labels == c], axis=0)) ** 2)

    return between / (within + 1e-9)  # Avoid div by 0

def mean_centroid_distance(embedding, labels):
    from scipy.spatial.distance import pdist
    labels = np.array(labels)
    classes = np.unique(labels)
    centroids = [np.mean(embedding[labels == c], axis=0) for c in classes]
    return np.mean(pdist(centroids))

def silhouette_score_metric(embedding, labels):
    from sklearn.metrics import silhouette_score
    return silhouette_score(embedding, labels)


def evaluate_significance_after_PCA(
	mouse_ids,
	which_feature,
	seq_types,
	frequencies,
	key_to_pat_dict,
	sessions_by_id,
	data,
	make_data_fn,
	num_components=10,
	num_shuffles=100,
	metric_fn=None,
	seed=0):

	if metric_fn is None:
		metric_fn = lambda emb, labels: emb[:, 0].var()  # default: PC1 variance

	rng = np.random.default_rng(seed)

	def run_pipeline(shuffle=False):
		reduced_all = []
		all_labels = []

		for mouse_id in mouse_ids:
			if which_feature == 'seqtype':
				labels = [key_to_pat_dict[st] for st in seq_types]
				data_pseudo = make_data_fn(labels, mouse_id, sessions_by_id, data, shuffle_seqtypes=shuffle)
			elif which_feature == 'frequency':
				labels = frequencies
				data_pseudo = make_data_fn(frequencies, mouse_id, sessions_by_id, data, pseudopop=True, shuffle_freqs=shuffle)
			else:
				raise ValueError("which_feature must be 'seqtype' or 'frequency'")

			mean_activity = np.mean(data_pseudo, axis=1)
			pca = PCA(n_components=num_components)
			reduced = pca.fit_transform(mean_activity)

			reduced_all.append(reduced)
			all_labels.extend(labels)  # replicate across mice

		# Align
		reference = reduced_all[0]
		aligned_all = [reference]

		for r in reduced_all[1:]:
			R, _ = orthogonal_procrustes(r, reference)
			aligned_all.append(r @ R)

		embedding_all = np.vstack(aligned_all)
		return metric_fn(embedding_all, all_labels)

	metric_real = run_pipeline(shuffle=False)
	metric_shuffled = [run_pipeline(shuffle=True) for _ in range(num_shuffles)]

	fig = plt.figure(figsize=(6, 4))
	plt.hist(metric_shuffled, bins=30, alpha=0.7, label='Shuffled')
	plt.axvline(metric_real, color='red', linestyle='--', label='Real')
	plt.xlabel(f"{metric_fn}")
	plt.ylabel("Count")
	plt.title(f"Real vs. Shuffled ({which_feature})")
	plt.legend()
	plt.tight_layout()
	fig.savefig(f"figs/{which_feature}_{metric_fn}.svg", bbox_inches='tight')


	return metric_real, metric_shuffled


if __name__ == "__main__":

	filename = join('full_patt_dict_ABCD_vs_ABBA.pkl')

	which_feature = 'seqtype'  # options: 'seqtype', 'frequency
		
	with open(filename, 'rb') as handle:
		data = pickle.load(handle)

	seq_types =['A-0', 'A-1', 'A-2', 'A-3']

	key_to_pat_dict = {'A-0': 'ABCD0', 'A-1': 'ABBA0', 'A-2': 'ABCD1', 'A-3':'ABBA1'}
	
	frequencies = ['A0', 'B0', 'C0', 'D0', 'A1', 'B1', 'C1', 'D1']  # frequency types for ABCD at lower pitch

	# extracts neural data for a single mouse for a single session for a single sequence pattern type (ABCD at lower pitch)

	mouse_sess_entries = list(data.keys())

	# Extract identities using a list comprehension
	mouse_ids = [entry.split('_')[0] for entry in mouse_sess_entries]

	# Dictionary to store sessions by identity
	sessions_by_id = defaultdict(list)

	for entry in mouse_sess_entries:
		identity, session = entry.split('_')
		sessions_by_id[identity].append(session)
	
	data = rename_dict_keys(data, seq_types, mouse_ids, sessions_by_id, key_to_pat_dict)

	n_components = 10

	unique_mouse_ids = sorted(set(mouse_ids))
	mouse_ids = [[mouse_id] for mouse_id in unique_mouse_ids] #+ [unique_mouse_ids]
	fignames = np.append(unique_mouse_ids, 'all')

	# print('RUNNING PCA ACROSS TIME')
	# run_PCA_across_time(
	# 	mouse_ids=mouse_ids,
	# 	fignames=fignames,
	# 	which_feature=which_feature,  
	# 	seq_types=seq_types,
	# 	frequencies=frequencies,
	# 	key_to_pat_dict=key_to_pat_dict,
	# 	sessions_by_id=sessions_by_id,
	# 	data=data,
	# 	n_components=n_components)  

			
	# print('RUNNING PLOT PCA BY SESSION')
	# plot_PCA_by_session(
	# 	unique_mouse_ids=unique_mouse_ids,
	# 	fignames=fignames,
	# 	frequencies=frequencies,
	# 	sessions_by_id=sessions_by_id,
	# 	data=data,
	# 	n_components=2,  
	# 	output_prefix='pca_freq_scatter')

	# print('PLOTTING FIRING RATES BY FREQUENCY OR SEQTYPE')
	# plot_firing_rate(mouse_ids, fignames, frequencies, n_components, which_feature, key_to_pat_dict, sessions_by_id, data)

	print('RUNNING PLOT ALIGNED PCA COMPS')
	plot_aligned_PCA_components(
		mouse_ids=mouse_ids,
		fignames=fignames,
		which_feature=which_feature,  
		seq_types=seq_types,
		frequencies=frequencies,
		key_to_pat_dict=key_to_pat_dict,
		sessions_by_id=sessions_by_id,
		data=data,
		num_components=4)

	exit()
	for metric_fn in [silhouette_score_metric, between_within_variance_ratio, mean_centroid_distance]:
		print(f'EVALUATING SIGNIFICANCE AFTER PCA with metric: {metric_fn.__name__}')
		# Evaluate significance after PCA
		# This will return the real metric and a list of shuffled metrics
		# The shuffles are done by shuffling the sequence types or frequencies
		# depending on which_feature
		# The metric_fn is a function that takes the embedding and labels and returns a single
		# value that quantifies the separation of the sequence types or frequencies
		metric_real, metric_shuffled = evaluate_significance_after_PCA(
			mouse_ids,
			which_feature='frequency',
			seq_types=seq_types,
			frequencies=frequencies,
			key_to_pat_dict=key_to_pat_dict,
			sessions_by_id=sessions_by_id,
			data=data,
			make_data_fn=make_data_freqs,
			metric_fn=metric_fn,
			num_components=4,
			num_shuffles=300)

	# filename = join('aligned_pupil_epochs.pkl')

	# data_by_seq = defaultdict(list)
	# print(mouse_ids)
	# exit()
	# for mouse_id in mouse_ids:
	# 	for session in sessions_by_id[mouse_id[0]]:
	# 		key = f'{mouse_id[0]}_{session}'
	# 		print(key)
	# 		if key not in data:
	# 			continue

	# 		# Get available sequence types for this session
	# 		available_seqtypes = [key_to_pat_dict[st] for st in seq_types] 
	# 		print(available_seqtypes)
	# 		D = []
	# 		stimulus = []
	# 		trial = []

	# 		for seqtype in available_seqtypes[:2]:
	# 			print(seqtype)

	# 			data_single_sess = data[key][seqtype]  # (trials, neurons, timepoints)
	# 			data_single_sess = data_single_sess.transpose(0, 2, 1)  # (trials, timepoints, neurons)
	# 			num_trials = np.shape(data_single_sess)[0]
	# 			num_timepoints = np.shape(data_single_sess)[1]
	# 			data_single_sess = data_single_sess.reshape(-1, np.shape(data_single_sess)[-1])  # (trials, timepoints, neurons)
	# 			D = np.append(D, data_single_sess, axis=0) if len(D) > 0 else data_single_sess


	# 			trial = np.append(trial, [np.repeat(t, num_timepoints) for t in range(num_trials) ]	)

	# 			stimulus = np.append(stimulus, np.repeat(seqtype, np.shape(data_single_sess)[0]))

	# 		data_dict = {
	# 			'raster': D,
	# 			'stimulus': stimulus,
	# 			'trial': trial
	# 		}

	# 		conditions = {'stimulus': available_seqtypes[:2]}

	# 		dec = Decodanda(data=data_dict, conditions=conditions)

	# 		performances, null = dec.decode(
    #                     training_fraction=0.5,  # fraction of trials used for training
    #                     cross_validations=10,   # number of cross validation folds
    #                     nshuffles=20,           # number of null model iterations
	# 					plot=True)
			
			
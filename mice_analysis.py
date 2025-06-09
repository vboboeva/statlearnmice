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

def make_pseudopop_seqtype(seq_types, mouse_ids, sessions_by_id, data):
	# make a pseudopopulation of neural data by averaging across trials for each mouse and session
	# and stacking them together: diff sessions diff neurons
	# data is a dictionary with keys as 'mouse_id_session' and values as a dictionary with keys as sequence types
	# and values as neural data (num_trials, num_neurons, num_timepts)
	# seq_types is a list of sequence types (e.g. ['A-0', 'A-1', 'A-2', 'A-3'])
	# mouse_ids is a list of mouse ids to run analysis on
	which_zscore = 'across_time'  # options: 'across_trials_and_time', 'across_trials', 'across_time'
	data_pseudopop_temp = []
	for seq_type in seq_types:
		# print(seq_type)
		for mouse_id in mouse_ids:
			# print(mouse_id)
			for session in sessions_by_id[mouse_id]:
				# print(session)
				key = f'{mouse_id}' + '_' + f'{session}'
				data_single_sess = data[key][seq_type] 	
				num_trials, num_neurons, num_timepts = data_single_sess.shape
				# print(data_single_sess.shape)
				if which_zscore == 'across_trials_and_time':
					data_single_sess = data_single_sess.transpose(0, 2, 1)  # shape (num_trials, num_timepts, num_neurons)
					data_single_sess = data_single_sess.reshape(-1, num_neurons)
					data_single_sess = zscore(data_single_sess, axis=0)  # z-score across neurons
					data_single_sess = data_single_sess.reshape(num_trials, num_timepts, num_neurons)
					data_single_sess = data_single_sess.transpose(0, 2, 1)  # reshape (num_trials, num_neurons, num_timepts)
				elif which_zscore == 'across_trials':
					data_single_sess = zscore(data_single_sess, axis=0)  # z-score across neurons
				elif which_zscore == 'across_time':
					data_single_sess = zscore(data_single_sess, axis=2)

				data_single_sess[np.isnan(data_single_sess)] = 0.0
				data_single_sess_avg_across_trials = np.nanmean(data_single_sess, axis=0)
				data_pseudopop_temp.append(data_single_sess_avg_across_trials)

	data_pseudopop_stacked = np.vstack(data_pseudopop_temp)
	assert data_pseudopop_stacked.shape[0] % len(seq_types) == 0

	data_pseudopop = data_pseudopop_stacked.reshape(len(seq_types), data_pseudopop_stacked.shape[0] // 4, data_pseudopop_stacked.shape[1])
	data_pseudopop = data_pseudopop.transpose(0,2,1) # putting N at the last dimension
	return data_pseudopop



def make_pseudopop_freqs(frequencies, mouse_ids, sessions_by_id, data):
	# make a new dictionary
	which_zscore = 'across_time'  # options: 'across_trials_and_time', 'across_trials', 'across_time'
	data_new = {}
	for mouse_id in mouse_ids:
		# print(mouse_id)
		for idx_sess, session in enumerate(sessions_by_id[mouse_id]):
			# print(session)
			key = f'{mouse_id}' + '_' + f'{session}'

			data_new[key] = {'A0':[], 'B0':[], 'C0':[], 'D0':[]}

			for seq_type in ['ABCD0', 'ABBA0']:
				# print(seq_type)
				data_single_sess = data[key][seq_type] 	
				num_trials, num_neurons, num_timepts = data_single_sess.shape

				if seq_type == 'ABCD0':
					data_new[key]['A0'] = data_single_sess[:, :, 25:40]
					data_new[key]['B0'] = data_single_sess[:, :, 50:65]
					data_new[key]['C0'] = data_single_sess[:, :, 75:90]
					data_new[key]['D0'] = data_single_sess[:, :, 100:115]
					
				elif seq_type == 'ABBA0':
					data_new[key]['A0'] = np.concatenate((data_new[key]['A0'], data_single_sess[:, :, 25:40]), axis=0)
					data_new[key]['B0'] = np.concatenate((data_new[key]['B0'], data_single_sess[:, :, 50:65]), axis=0)
					data_new[key]['B0'] = np.concatenate((data_new[key]['B0'], data_single_sess[:, :, 75:90]), axis=0)
					data_new[key]['A0'] = np.concatenate((data_new[key]['A0'], data_single_sess[:, :, 100:115]), axis=0)
				
			data_new[key]['A0'] = np.array(data_new[key]['A0'])
			data_new[key]['B0'] = np.array(data_new[key]['B0'])
			data_new[key]['C0'] = np.array(data_new[key]['C0'])
			data_new[key]['D0'] = np.array(data_new[key]['D0'])

	data_pseudopop_temp = []
	for frequency in frequencies:
		# print(frequency)
		count=0
		for mouse_id in mouse_ids:
			# print(mouse_id)
			for session in sessions_by_id[mouse_id]:
				# print(session)
				key = f'{mouse_id}' + '_' + f'{session}'
				data_single_sess = data_new[key][frequency]
				# print(np.shape(data_single_sess))
				num_trials, num_neurons, num_timepts = data_single_sess.shape
				if which_zscore == 'across_trials_and_time':
					data_single_sess = data_single_sess.transpose(0, 2, 1)  # shape (num_trials, num_timepts, num_neurons)
					data_single_sess = data_single_sess.reshape(-1, num_neurons)
					data_single_sess = zscore(data_single_sess, axis=0)  # z-score across neurons
					data_single_sess = data_single_sess.reshape(num_trials, num_timepts, num_neurons)
					data_single_sess = data_single_sess.transpose(0, 2, 1)  # reshape (num_trials, num_neurons, num_timepts)
				elif which_zscore == 'across_trials':
					data_single_sess = zscore(data_single_sess, axis=0)
				elif which_zscore == 'across_time':
					data_single_sess = zscore(data_single_sess, axis=2)

				data_single_sess[np.isnan(data_single_sess)] = 0.0
				data_single_sess_avg_across_trials = np.nanmean(data_single_sess, axis=0)
				data_pseudopop_temp.append(data_single_sess_avg_across_trials)

	data_pseudopop_stacked = np.vstack(data_pseudopop_temp)
	assert data_pseudopop_stacked.shape[0] % len(frequencies) == 0

	data_pseudopop = data_pseudopop_stacked.reshape(len(frequencies), data_pseudopop_stacked.shape[0] // 4, data_pseudopop_stacked.shape[1])
	data_pseudopop = data_pseudopop.transpose(0,2,1) # putting N at the last dimension

	return data_pseudopop

def plot_PCA(data_embedding, seq_types, key_to_pat_dict, n_components, frac_variance, figname, which_feature='seqtype'):
	fig, ax = plt.subplots(2,5, figsize=(20, 8))
	ax = ax.flatten()
	colors = ['Blue', 'Purple', 'Green', 'Red']

	for comp in range(n_components):
		for i in range(len(seq_types)):
			ax[comp].plot(np.arange(np.shape(data_embedding)[1]), data_embedding[i,:,comp], color=colors[i], label=key_to_pat_dict[seq_types[i]])
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
	fig.suptitle(f'{_N} neurons across sessions')
	fig.savefig(f'pca_components_{figname}_{which_feature}.svg', bbox_inches='tight')	


def plot_3D_PCA_trajectory(data_embedding, figname, which_feature='seqtype'):
    # data_embedding: shape (n_patterns, n_timepoints, n_components)

    colormaps = [cm.Blues, cm.Purples, cm.Greens, cm.Reds]
    pc_triplets = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    n_triplets = len(pc_triplets)

    fig = plt.figure(figsize=(8 * n_triplets, 6))
    for idx, (pcx, pcy, pcz) in enumerate(pc_triplets):
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
    fig.savefig(f"PCA_3D_triplets_{figname}_{which_feature}.svg", bbox_inches="tight", dpi=600)


def bandstop_filter(data, lowcut=4, highcut=12, fs=100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data)


def plot_bandpass(raw_signals, figname, freq_types, n_components, which_feature='seqtype'):
	fig, ax = plt.subplots(3, 4, figsize=(20, 10))

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

			ax[0,i].plot(np.arange(len(signal)), signal, label='Raw Signal', color='gray', alpha=0.1)

			# Plot filtered signal
			if which_feature == 'seqtype':
				filtered = bandstop_filter(signal, fs=fs)
				filtered_signal[i, :, j] = filtered
				ax[1,i].plot(np.arange(len(signal)), filtered, label='Filtered Signal (Theta Band)', ls='--', color='gray', alpha=0.1)

			# PSD (Power Spectral Density)
			frequencies, power = welch(signal, nperseg=64, fs=fs)
			ax[2,i].semilogy(frequencies, power, color='gray', alpha=0.1)

		if which_feature == 'seqtype':
			ax[0,i].set_title(f'Seq Type: {key_to_pat_dict[seq_types[i]]}', fontsize=16)
			ax[1,i].set_title(f'Seq Type: {key_to_pat_dict[seq_types[i]]}', fontsize=16)
		else:
			ax[0,i].set_title(f'Freq Type: {freq_types[i]}', fontsize=16)
			ax[1,i].set_title(f'Freq Type: {freq_types[i]}', fontsize=16)

		ax[0,i].set_xlabel('Time (s)')
		ax[0,i].set_ylabel('Amplitude')
		ax[0,i].set_xticks(np.arange(0, len(signal), 15))
		ax[0,i].set_xticklabels(np.arange(0, len(signal)*10, 15*10))
		
		ax[1,i].set_xlabel('Time (s)')
		ax[1,i].set_ylabel('Amplitude')
		ax[1,i].set_xticks(np.arange(0, len(signal), 15))
		ax[1,i].set_xticklabels(np.arange(0, len(signal)*10, 15*10))

		ax[2,i].set_xlabel('Frequency (Hz)')
		ax[2,i].set_ylabel('Power')
		ax[2,i].axvspan(4, 12, color='gray', alpha=0.3)

		ax[0,i].plot(np.arange(len(signal)), np.mean(raw_signal, axis=1), label='Raw Signal', color='black')
		ax_inset = inset_axes(ax[0,i], width="40%", height="30%", loc=1)  # loc=1 is upper right
		ax_inset.plot(np.arange(len(signal)), np.mean(raw_signal, axis=1), label='Raw Signal', color='black')
		
		ax[1,i].plot(np.arange(len(signal)), np.mean(filtered_signal[i], axis=1), label='Filtered Signal', color='black')
		ax_inset = inset_axes(ax[1,i], width="40%", height="30%", loc=1)  # loc=1 is upper right
		ax_inset.plot(np.arange(len(signal)), np.mean(filtered_signal[i], axis=1), label='Raw Signal', color='black')

	fig.tight_layout()
	fig.savefig(f"filter_{figname}_{which_feature}.svg", bbox_inches="tight", dpi=600)
	
	return filtered_signal

if __name__ == "__main__":

	filename = join('full_patt_dict_ABCD_vs_ABBA.pkl')

	which_feature = 'seqtype'  # options: 'seqtype', 'frequency
		
	with open(filename, 'rb') as handle:
		data = pickle.load(handle)

	seq_types=['A-0', 'A-1', 'A-2', 'A-3']

	key_to_pat_dict= {'A-0': 'ABCD0', 'A-1': 'ABBA0', 'A-2': 'ABCD1', 'A-3':'ABBA1'}
	
	freq_types = ['A0', 'B0', 'C0', 'D0']  # frequency types for ABCD at lower pitch

	# extracts neural data for a single mouse for a single session for a single sequence pattern type (ABCD at lower pitch)

	mouse_sess_entries = list(data.keys())

	# Extract identities using a list comprehension
	mouse_ids = [entry.split('_')[0] for entry in mouse_sess_entries]

	# Optionally, get unique identities
	unique_mouse_ids = sorted(set(mouse_ids))

	# Dictionary to store sessions by identity
	sessions_by_id = defaultdict(list)

	for entry in mouse_sess_entries:
		identity, session = entry.split('_')
		sessions_by_id[identity].append(session)
	
	data = rename_dict_keys(data, seq_types, mouse_ids, sessions_by_id, key_to_pat_dict)

	n_components = 10

	x = np.append([[unique_mouse_id] for unique_mouse_id in unique_mouse_ids], unique_mouse_ids)

	mouse_ids = [[x] for x in unique_mouse_ids] + [['DO79', 'D080', 'DO81']]
	fignames = np.append(unique_mouse_ids, 'all_mice')
	print(mouse_ids)
	print(fignames)

	for mouse_id, figname in zip(mouse_ids, fignames):
		print(mouse_id)

		if which_feature == 'seqtype':
			data_pseudopop = make_pseudopop_seqtype([key_to_pat_dict[st] for st in seq_types], mouse_id, sessions_by_id, data)
		elif which_feature == 'frequency':
			data_pseudopop = make_pseudopop_freqs(freq_types, mouse_id, sessions_by_id, data)
		else:
			raise ValueError("Invalid feature type. Choose 'seqtype' or 'frequency'.")

		_p, _t, _N = data_pseudopop.shape

		print('number of seqtypes=',_p)
		print('number of timepts=',_t)
		print('number of neurons',_N)

		print(data_pseudopop.reshape(-1, _N).shape)
		pca = PCA(n_components = n_components)
		# find principal components for a particular position in sequence 
		_ = pca.fit_transform(data_pseudopop.reshape(-1, _N))
		# project all sequences along them
		data_embedding = pca.transform(data_pseudopop.reshape(-1, _N))
		# print('data_embedding', data_embedding)
		# print('shape data_pseudopop', np.shape(data_pseudopop))

		proj_variance = np.cumsum(np.var(data_embedding, axis=0))
		total_variance = np.sum(np.var(data_pseudopop.reshape(-1, _N), axis=0))
		frac_variance = proj_variance / total_variance*np.ones(len(proj_variance))

		print("proj_variance =", np.shape(proj_variance))
		print("total_variance =", np.shape(total_variance))
		print("frac_variance =", frac_variance)

		data_embedding = data_embedding.reshape(_p, _t, n_components)

		signal = data_embedding  # choose btw data_embedding and data_pseudopop and set n_components accordingly
		filtered_signals = plot_bandpass(signal, figname, freq_types, n_components = n_components, which_feature=which_feature)		

		plot_PCA(data_embedding, seq_types, key_to_pat_dict, n_components, frac_variance, figname, which_feature=which_feature)
		plot_3D_PCA_trajectory(data_embedding, figname, which_feature=which_feature)

		
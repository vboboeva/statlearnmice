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

def make_pseudopop(seq_types, mouse_ids, sessions_by_id, data):
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

	threshold = 1e-10  # your threshold

	# Step 1: Get max activity for each neuron across all trialtypes and timepoints
	max_activity_per_neuron = data_pseudopop.max(axis=(0,1))  # shape: (N,)

	# Step 2: Find neurons that exceed the threshold at least once
	neurons_to_keep = max_activity_per_neuron > threshold  # boolean mask of shape (N,)

	# Step 3: Filter the original array to keep only those neurons
	filtered_activity = data_pseudopop[:,:,neurons_to_keep]

	print(f"Kept {neurons_to_keep.sum()} out of {data_pseudopop.shape[2]} neurons")
	# exit()
	return data_pseudopop

def plot_PCA(data_embedding, seq_types, key_to_pat_dict, n_components, frac_variance, figname):
		fig, ax = plt.subplots(2,5, figsize=(20, 8))
		ax = ax.flatten()
		colors = ['Blue', 'Purple', 'Green', 'Red']

		for comp in range(n_components):
			for i in range(len(seq_types)):
				ax[comp].plot(np.arange(np.shape(data_embedding)[1]), data_embedding[i,:,comp], color=colors[i], label=key_to_pat_dict[seq_types[i]])
				ax[comp].axvspan(0+25, 15+25, color='red', lw=0, alpha=0.03)
				ax[comp].axvspan(25+25, 40+25, color='red', lw=0, alpha=0.03)
				ax[comp].axvspan(50+25, 65+25, color='red', lw=0, alpha=0.03)
				ax[comp].axvspan(75+25, 90+25, color='red', lw=0, alpha=0.03)
				ax[comp].set_xticks(np.arange(0, 126, 20))
				ax[comp].set_xticklabels(np.arange(0, 1260, 200))
				ax[comp].set_title(f'PC {comp+1} ({frac_variance[comp]:.2f} variance)', fontsize=16)
				# ax[comp].set_ylim(-4, 5)
		ax[5].set_xlabel('Time (ms)')
		ax[5].set_ylabel('Component Value')
		ax[0].legend()
		fig.suptitle(f'{_N} neurons across sessions')
		fig.savefig(f'pca_components_ABCD_vs_ABBA_{figname}.png', bbox_inches='tight')	


def plot_3D_PCA_trajectory(data_embedding, figname):
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')

	colormaps = [cm.Blues, cm.Purples, cm.Greens, cm.Reds]
	# Set the viewing angle for better visualization
	for patt_idx, Y in enumerate(data_embedding):
		x, y, z = Y[:, 0], Y[:, 1], Y[:, 2]

		# Create line segments between consecutive time points
		points = np.array([x, y, z]).T  # shape (T, 3)
		segments = np.array([[points[i], points[i+1]] for i in range(len(points) - 1)])

		# Normalize time for colormap (0=early, 1=late)
		norm_time = np.linspace(0.2, 0.8, len(segments))
		colors = colormaps[patt_idx](norm_time)  # colormap: light to dark

		# Create a 3D line collection with time-colored segments
		lc = Line3DCollection(segments, colors=colors, linewidth=2)
		ax.add_collection3d(lc)

	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.set_zlabel('PC3')
	ax.set_title('3D PCA Trajectories (Color = Time)')

	# Autoscale axes to fit data
	all_xyz = np.vstack([Y[:, :3] for Y in data_embedding])
	ax.auto_scale_xyz(all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2])

	fig.tight_layout()
	fig.savefig(f"PCA_3D_{figname}.svg", bbox_inches="tight", dpi=600)


def bandstop_filter(data, lowcut=1, highcut=30, fs=100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data)


def plot_bandpass(raw_signals, figname, n_components):
	fig, ax = plt.subplots(3, 4, figsize=(20, 10))

	colors = ['Blue', 'Purple', 'Green', 'Red']
	filtered_signal = np.zeros_like(raw_signals)
	
	for i, raw_signal in enumerate(raw_signals):

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
			filtered = bandstop_filter(signal, fs=100)
			filtered_signal[i, :, j] = filtered
			# Plot filtered signal
			ax[0,i].plot(np.arange(len(signal)), signal, label='Raw Signal', color='gray', alpha=0.1)

			ax[1,i].plot(np.arange(len(signal)), filtered, label='Filtered Signal (Theta Band)', ls='--', color='gray', alpha=0.1)

			# PSD (Power Spectral Density)
			frequencies, power = welch(signal, nperseg=64, fs=100)
			ax[2,i].semilogy(frequencies, power, color='gray', alpha=0.1)

		ax[0,i].set_title(f'Seq Type: {key_to_pat_dict[seq_types[i]]}', fontsize=16)
		ax[0,i].set_xlabel('Time (s)')
		ax[0,i].set_ylabel('Amplitude')
		ax[0,i].set_xticks(np.arange(0, 126, 20))
		ax[0,i].set_xticklabels(np.arange(0, 1260, 200))
		
		ax[1,i].set_title(f'Seq Type: {key_to_pat_dict[seq_types[i]]}', fontsize=16)
		ax[1,i].set_xlabel('Time (s)')
		ax[1,i].set_ylabel('Amplitude')
		ax[1,i].set_xticks(np.arange(0, 126, 20))
		ax[1,i].set_xticklabels(np.arange(0, 1260, 200))

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
	fig.savefig(f"filter_{figname}.png", bbox_inches="tight", dpi=600)
	
	return filtered_signal

if __name__ == "__main__":

	filename = join('full_patt_dict_ABCD_vs_ABBA.pkl')
		
	with open(filename, 'rb') as handle:
		data = pickle.load(handle)
	
	# print(data.keys())
	key_to_pat_dict= {'A-0': 'ABCD0', 'A-1': 'ABBA0', 'A-2': 'ABCD1', 'A-3':'ABBA1'}
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
	
	seq_types=['A-0', 'A-1', 'A-2', 'A-3']

	n_components = 10

	x = np.append([[unique_mouse_id] for unique_mouse_id in unique_mouse_ids], unique_mouse_ids)

	mouse_list = [[x] for x in unique_mouse_ids] + [['DO79', 'DO81', 'DO82']]
	fignames = np.append(unique_mouse_ids, 'all_mice')
	print(mouse_list)
	print(fignames)

	for mouse_id, figname in zip(mouse_list, fignames):
		print(mouse_id)
		data_pseudopop = make_pseudopop(seq_types, mouse_id, sessions_by_id, data)

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
		filtered_signals = plot_bandpass(signal, figname, n_components = n_components)

		plot_PCA(filtered_signals, seq_types, key_to_pat_dict, n_components, frac_variance, figname)
		plot_3D_PCA_trajectory(filtered_signals, figname)

		# exit()

		# exit()


			



	


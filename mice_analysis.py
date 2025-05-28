import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.decomposition import PCA
import pickle
from os.path import join
from collections import defaultdict
import scipy
from scipy.stats import zscore

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
		print(seq_type)
		for mouse_id in mouse_ids:
			print(mouse_id)
			for session in sessions_by_id[mouse_id]:
				print(session)
				key = f'{mouse_id}' + '_' + f'{session}'
				data_single_sess = data[key][seq_type] 	
				num_trials, num_neurons, num_timepts = data_single_sess.shape
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

				data_single_sess[np.isnan(data_single_sess)] = 0.000
				data_single_sess_avg_across_trials = np.nanmean(data_single_sess, axis=0)
				data_pseudopop_temp.append(data_single_sess_avg_across_trials)

	data_pseudopop_stacked = np.vstack(data_pseudopop_temp)
	print(np.shape(data_pseudopop_stacked))
	print(data_pseudopop_stacked)
	exit()

	assert data_pseudopop_stacked.shape[0] % len(seq_types) == 0

	data_pseudopop = data_pseudopop_stacked.reshape(len(seq_types), data_pseudopop_stacked.shape[0] // 4, data_pseudopop_stacked.shape[1])
	data_pseudopop = data_pseudopop.transpose(0,2,1) # putting N at the last dimension


	return data_pseudopop	

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

	mouse_list = [[x] for x in unique_mouse_ids] + [unique_mouse_ids]
	fignames = np.append(unique_mouse_ids, 'all_mice')
	print(mouse_list)
	# exit()
	print(fignames)

	for mouse_id, figname in zip(mouse_list, fignames):
		print(mouse_id)
		data_pseudopop = make_pseudopop(seq_types, mouse_id, sessions_by_id, data)

		_p, _t, _N = data_pseudopop.shape
		print('number of seqtypes=',_p)
		print('number of timepts=',_t)
		print('number of neurons',_N)

		pca = PCA(n_components = n_components)
		# find principal components for a particular position in sequence 
		_ = pca.fit_transform(data_pseudopop.reshape(-1, _N))
		# project all sequences along them
		data_embedding = pca.transform(data_pseudopop.reshape(-1, _N))
		print('shape data_embedding', np.shape(data_embedding))
		print('shape data_pseudopop', np.shape(data_pseudopop))

		proj_variance = np.cumsum(np.var(data_embedding, axis=0))
		total_variance = np.sum(np.var(data_pseudopop.reshape(-1, _N), axis=0))
		frac_variance = proj_variance / total_variance*np.ones(len(proj_variance))

		print("proj_variance =", np.shape(proj_variance))
		print("total_variance =", np.shape(total_variance))
		print("frac_variance =", frac_variance)

		data_embedding = data_embedding.reshape(_p, _t, n_components)

		fig, ax = plt.subplots(2,5, figsize=(20, 10))
		ax = ax.flatten()

		for comp in range(n_components):
			ax[comp].set_xlabel('Time')
			ax[comp].set_ylabel('Component Value')
			for i in range(len(seq_types)):
				ax[comp].plot(np.arange(np.shape(data_embedding)[1]), data_embedding[i,:,comp], label=key_to_pat_dict[seq_types[i]])
				ax[comp].axvspan(0+25, 15+25, color='red', lw=0, alpha=0.05)
				ax[comp].axvspan(25+25, 40+25, color='red', lw=0, alpha=0.05)
				ax[comp].axvspan(50+25, 65+25, color='red', lw=0, alpha=0.05)
				ax[comp].axvspan(75+25, 90+25, color='red', lw=0, alpha=0.05)
				ax[comp].set_title(f'PC {comp+1} ({frac_variance[comp]:.2f} variance)', fontsize=16)
			
		ax[0].legend()
		fig.suptitle(f'{_N} neurons across sessions')
		fig.savefig(f'pca_components_ABCD_vs_ABBA_{figname}.svg', bbox_inches='tight')

			



	


import os
import sys
import warnings
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from imblearn.under_sampling import OneSidedSelection, NeighbourhoodCleaningRule
import h5py as h5

if __name__ == '__main__':

 usecols = [0, 1, 2, 3, 4, 5,6, 7, 8,9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
 #usecols = ['TSS','ATF2', 'BACH1', 'JUN', 'CMYC',	'E2F6',	'FOSL1','NANOG','NRSF',	'POU5F1', 'SIN3A', 'SP4','TCF12','TEAD4','ATF3','CEBPB','CREB','CTCF','EGR1','GABP','JUND','MAFK','MAX','RFX5',	'SIX5','SP1','SRF','USF1','USF2','YY1','ZNF274']
 dtype = {usecols[0]: np.str, usecols[1]: np.float64, usecols[2]: np.float64, usecols[3]: np.float64, usecols[4]: np.float64, usecols[5]: np.float64, usecols[6]: np.float64,usecols[7]: np.float64, usecols[8]: np.float64,
			usecols[9]: np.float64, usecols[10]: np.float64, usecols[11]: np.float64, usecols[12]: np.float64, usecols[13]: np.float64, usecols[14]: np.float64, usecols[15]: np.float64,usecols[16]: np.float64,usecols[17]: np.float64, 
			usecols[18]: np.float64, usecols[19]: np.float64, usecols[20]:  np.float64, usecols[21]: np.float64, usecols[22]: np.float64, usecols[23]: np.float64, usecols[24]: np.float64, usecols[25]: np.float64, usecols[26]: np.float64,
			usecols[27]: np.float64, usecols[28]: np.float64, usecols[29]: np.float64, usecols[30]: np.float64}
 skiprows = 1 
 input_RPKM = pd.read_csv('~/code/data/H1_TSS_TF/H1_TSS_TF_Normalised_RPKMs.txt', sep='\t', header=None, usecols=usecols, dtype=dtype, skiprows=skiprows)
 input_features = input_RPKM.values
 X = input_features[:,1:31]
 print(X.shape)
 
 usecols = [0, 1, 2, 3, 4, 5]
 dtype = {usecols[0]: np.str, usecols[1]: np.int8, usecols[2]: np.int8, usecols[3]: np.int8, usecols[4]: np.int8, usecols[5]: np.int8}
 skiprows = 1
 out_label = pd.read_table('~/code/data/features/label_int.txt', header=None, usecols=usecols, dtype=dtype, skiprows=skiprows)
 output_H3K4me3 = out_label.values[:,4]
 y = np.array(output_H3K4me3, dtype='int8')
 # y_reshape = y.reshape(len(y),1)
 
 # Instanciate a PCA object for the sake of easy visualisation
 pca = PCA(n_components=2)
 
 # Fit and transform x to visualise inside a 2D feature space
 X_vis= pca.fit_transform(X)
 
 # Apply NeighbourhoodCleaningRule
 ncl = NeighbourhoodCleaningRule(random_state = 42, return_indices=True)
 X_resampled, y_resampled, idx_resampled = ncl.fit_sample(X, y)
 X_res_vis = pca.transform(X_resampled)
 
 fig = plt.figure()
 ax = fig.add_subplot(1, 1, 1)
 idx_samples_removed = np.setdiff1d(np.arange(X_vis.shape[0]), idx_resampled)
 
 frq = np.bincount(y_resampled)
 aar_neg = np.transpose((y_resampled==0).nonzero()) 
 aar_pos = np.transpose((y_resampled==1).nonzero())
 idx_class_0 = y_resampled == 0
 
 h5filename = "histonemodTF_resample_ncl.h5"
 if os.path.exists(h5filename):
	os.remove(h5filename)
 h5file = h5.File(h5filename,'w')
 in_group = h5file.create_group('input')
 in_group.create_dataset('H3K4me3_RPKM',data = X_resampled, dtype = np.float64, compression ='gzip')
 out_group = h5file.create_group('output')
 out_group.create_dataset('H3K4me3',data = y_resampled, dtype = np.int8, compression ='gzip')	 
 h5file.close()
 
 print(X_resampled.shape)
 print(y_resampled.shape)
 #print(X_resampled[1:10,:])
 #print(y_resampled[1:10])
 print(aar_neg.shape)
 print(aar_pos.shape)	
 print(idx_samples_removed.shape)
 print(frq)
 
 plt.scatter(X_res_vis[idx_class_0, 0], X_res_vis[idx_class_0, 1], alpha=.8, label='Class #0')
 plt.scatter(X_res_vis[~idx_class_0, 0], X_res_vis[~idx_class_0, 1], alpha=.8, label='Class #1')
 plt.scatter(X_vis[idx_samples_removed, 0], X_vis[idx_samples_removed, 1], alpha=.8, label='Removed samples')

 # make nice plotting
 ax.spines['top'].set_visible(False)
 ax.spines['right'].set_visible(False)
 ax.get_xaxis().tick_bottom()
 ax.get_yaxis().tick_left()
 ax.spines['left'].set_position(('outward', 10))
 ax.spines['bottom'].set_position(('outward', 10))
 ax.set_xlim([-10, 10])
 ax.set_ylim([-8, 12])
 
 plt.title('Under-sampling using one-sided selection')
 plt.legend()
 plt.tight_layout()
 plt.show()
 
 

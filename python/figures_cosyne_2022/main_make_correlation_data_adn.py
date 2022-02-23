import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from matplotlib.gridspec import GridSpecFromSubplotSpec


mappings = pd.read_csv("/mnt/DataRAID/MergedData/MAPPING_NUCLEUS.csv", index_col = 0)
datasets = np.loadtxt('/mnt/DataRAID/MergedData/'+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
	swr_mod, swr_ses = loadSWRMod('/mnt/DataRAID/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
	nbins 		= 400
	binsize		= 5
	times 		= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	swr_mod 	= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
	swr_mod = swr_mod.drop(swr_mod.columns[swr_mod.isnull().any()].values, axis = 1)
	swr_mod = swr_mod.loc[-500:500]
	neurons = np.intersect1d(swr_mod.columns.values, mappings.index.values)
	hd_neurons = mappings.loc[neurons][mappings.loc[neurons, 'hd'] == 1].index.values
	
	return swr_mod[hd_neurons]

swr_mod_adn = loadadndatapapier()


lmn = np.where(shank == 9)[0]


cc_lmn = cc_rip2[lmn].loc[-500:500]

cc_lmn = cc_lmn.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2)

cc_rip = cc_rip.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2)

figure()
subplot(221)
plot(swr_mod_adn.mean(1), color = 'red')
subplot(223)
plot(cc_rip[lmn].loc[-500:500], color = 'green', alpha = 0.25)
plot(cc_rip[lmn].loc[-500:500].mean(1), color = 'green')
subplot(222)
plot(swr_mod_adn.mean(1), color= 'red')
plot(cc_lmn, color = 'green', alpha = 0.5)
plot(cc_lmn.mean(1), color = 'green', alpha = 0.5)
subplot(224)
plot(swr_mod_adn.mean(1), color = 'red')
plot(cc_lmn.mean(1), color = 'green')



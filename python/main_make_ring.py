import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
from umap import UMAP
import sys
from matplotlib.colors import hsv_to_rgb



path 								= '/mnt/DataGuillaume/LMN/A1407/A1407-190416'
# path 								= '../data/A1400/A1407/A1407-190416'

episodes = ['sleep', 'wake', 'sleep']
events = [1]

spikes, shank 						= loadSpikeData(path)
n_channels, fs, shank_to_channel 	= loadXML(path)
position 							= loadPosition(path, events, episodes)
wake_ep 							= loadEpoch(path, 'wake', episodes)
sleep_ep 							= loadEpoch(path, 'sleep')					


tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tokeep, stat 						= findHDCells(tuning_curves[1])

tcurves 							= tuning_curves[1][tokeep]
tcurves 							= smoothAngularTuningCurves(tcurves, 10, 2)
tcurves 							= tcurves[tcurves.columns[tcurves.idxmax().argsort().values]]


neurons 							= np.sort(list(spikes.keys()))[tokeep]

####################################################################################################################
# BIN WAKE
####################################################################################################################
bin_size = 50
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

rate = np.sqrt(spike_counts/(bin_size*1e-3))

angle = position['ry']
wakangle = pd.Series(index = np.arange(len(bins)-1))
tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
wakangle.loc[tmp.index] = tmp
wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)
H = wakangle.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

# sys.exit()


tmp = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1).values


ump = UMAP(n_neighbors = 500, min_dist = 1).fit_transform(tmp)
figure()
scatter(ump[:,0], ump[:,1], c= RGB, marker = '.', alpha = 0.5, linewidth = 0, s = 100)

show()




# from sklearn.manifold import Isomap
# imap = Isomap(n_neighbors = 100, n_components = 2).fit_transform(tmp)
# figure()
# scatter(imap[:,0], imap[:,1], c= RGB, marker = '.', alpha = 0.5, linewidth = 0, s = 100)






# figure()
# for i,n in enumerate(tcurves.columns):
# 	subplot(3,5,i+1, projection = 'polar')
# 	plot(tcurves[n])
 



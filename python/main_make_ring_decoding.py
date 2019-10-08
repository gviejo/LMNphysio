import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
from umap import UMAP
import sys
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D


# path 								= '/mnt/DataGuillaume/LMN/A1407/A1407-190416'
path 								= '../data/A1400/A1407/A1407-190416'

episodes = ['sleep', 'wake', 'sleep']
events = [1]

spikes, shank 						= loadSpikeData(path)
n_channels, fs, shank_to_channel 	= loadXML(path)
position 							= loadPosition(path, events, episodes)
wake_ep 							= loadEpoch(path, 'wake', episodes)
sleep_ep 							= loadEpoch(path, 'sleep')					

acceleration						= loadAuxiliary(path)
postsleep_ep						= refineSleepFromAccel(acceleration, sleep_ep.loc[[1]])


tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61, 400)
tokeep, stat 						= findHDCells(tuning_curves[1])

tcurves 							= tuning_curves[1][tokeep]
tcurves 							= smoothAngularTuningCurves(tcurves, 10, 2)
tcurves 							= tcurves[tcurves.columns[tcurves.idxmax().argsort().values]]


neurons 							= np.sort(list(spikes.keys()))[tokeep]

# neurons = np.array([0,7,1,4,10,6,2])

speed_curves = computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)

####################################################################################################################
# BIN WAKE
####################################################################################################################
bin_size = 400
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

rate_wake = np.sqrt(spike_counts/(bin_size*1e-3))
# rate_wake = spike_counts/(bin_size*1e-3)


# binning angle
angle = position['ry']
wakangle = pd.Series(index = np.arange(len(bins)-1))
tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
wakangle.loc[tmp.index] = tmp
wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

velocity	= np.diff(wakangle.values)/(bin_size/1000)


# resiz to match velocity
rate_wake = rate_wake.iloc[0:-1]
wakangle = wakangle.iloc[0:-1]

H = wakangle.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

vel = np.abs(velocity)
# SIZE = np.pi/(1+np.exp(-(vel - 4)*1))
SIZE = vel
# plot(vel, SIZE, 'o')

####################################################################################################################
# BIN SLEEP
####################################################################################################################
bin_size = 400
bins = np.arange(sleep_ep.loc[[1]].as_units('ms').start.iloc[0], sleep_ep.loc[[1]].as_units('ms').end.iloc[-1]+bin_size, bin_size)
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

rate_sleep = np.sqrt(spike_counts/(bin_size*1e-3))


####################################################################################################################
# PROJECTION
####################################################################################################################
tmp = rate_wake.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1).values
tmp2 = rate_sleep.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1).values
tmp3 = np.vstack((tmp, tmp2))


ump = UMAP(n_components = 2, n_neighbors = 20, min_dist = 1).fit_transform(tmp)

ump1 = ump[0:len(tmp)]
ump2 = ump[len(tmp):]


# ump2 = UMAP(n_components = 2, n_neighbors = 100, min_dist = 1).fit_transform(tmp2)


####################################################################################################################
# DECODING
####################################################################################################################
#center ring
# ump = ump - np.mean(ump,0)

# radius = 





figure()
scatter(ump1[:,0], ump1[:,1], s = 100, c= RGB, marker = '.', alpha = 0.8, linewidth = 0)


figure()
scatter(ump2[:,0], ump2[:,1], marker = '.', alpha = 0.5, linewidth = 0, s = 100)
show()


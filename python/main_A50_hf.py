import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from matplotlib.colors import hsv_to_rgb
from umap import UMAP

data_directory = '/mnt/DataGuillaume/LMN-ADN/A5002/A5002-200313A'

episodes = ['sleep', 'wake', 'wake', 'wake', 'sleep', 'wake', 'wake', 'sleep']

events = ['1', '2', '3', '5', '6']


spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
acceleration 						= acceleration[[0,1,2]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)




tcurves = {}
for i in wake_ep.index:
	tcurves[i] = computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[i]], 60)
	tcurves[i] = smoothAngularTuningCurves(tcurves[i], 10, 2)

tokeep = findHDCells(tcurves[0])[0]
tokeep = np.intersect1d(tokeep, np.where(shank == 3)[0])


############################################################################################### 
# PLOT
###############################################################################################

dataumap = {}

neurons = tokeep

figure()
count = 1
for k in wake_ep.index:
# for k in range(2):
	tmp_ep = nts.IntervalSet(start = wake_ep.loc[k,'start'], end = wake_ep.loc[k,'start'] + 20*60*1e6)
	bin_size = 400
	bins = np.arange(tmp_ep.as_units('ms').start.iloc[0], tmp_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)

	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
	for i in neurons:
		spks = spikes[i].as_units('ms').index.values
		spike_counts[i], _ = np.histogram(spks, bins)

	rate = np.sqrt(spike_counts/(bin_size*1e-3))

	angle = position['ry'].restrict(tmp_ep)
	wakangle = pd.Series(index = np.arange(len(bins)-1))
	tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
	wakangle.loc[tmp.loc[0:len(bins)-2].index] = tmp.loc[0:len(bins)-2]
	wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)
	H = wakangle.values/(2*np.pi)
	HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
	RGB = hsv_to_rgb(HSV)

	tmp = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1).values


	ump = UMAP(n_neighbors = 200, min_dist = 1).fit_transform(tmp)

	subplot(2,3,count)
	scatter(ump[:,0], ump[:,1], c= RGB, marker = 'o', alpha = 0.5, linewidth = 0, s = 100)

	dataumap[k] = {'ump': ump,
					'rgb': RGB}
	count += 1

show()


sys.exit()




############################################################################################### 
# PLOT
###############################################################################################
from matplotlib import gridspec
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

shank = shank.flatten()


# gs = grid

figure()
for k,i in enumerate(neurons):
	subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,k+1, projection = 'polar')
	for j in tcurves.keys():
	# for j in [0]:
		plot(tcurves[j][i], label = str(shank[i]) + ' ' + str(i))


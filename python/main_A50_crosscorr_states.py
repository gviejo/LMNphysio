import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean

# data_directory = '/mnt/DataGuillaume/LMN/A1410/A1410-200116A/A1410-200116A'
# data_directory = '/mnt/DataGuillaume/LMN-ADN/A5002/A5002-200303B'
data_directory = '/mnt/DataGuillaume/LMN/A1411/A1411-200908A'

episodes = ['sleep', 'wake', 'sleep']
# events = ['1', '3']
events = ['1']


spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
if 'A5002' in data_directory:
	acceleration 						= acceleration[[0,1,2]]
else:
	acceleration 						= acceleration[[3,4,5]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)


tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tcurves 							= smoothAngularTuningCurves(tuning_curves, 10, 2)
tokeep, stat 						= findHDCells(tcurves, z=10, p = 0.001)

# lmn = list(np.where(shank == 3)[0])
# lmn = np.intersect1d(lmn, tokeep)
# tokeep = lmn
tcurves 							= tuning_curves[tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]

neurons 							= [data_directory.split("/")[-1]+'_'+str(n) for n in tcurves.columns.values]

peaks.index							= pd.Index(neurons)
tcurves.columns						= pd.Index(neurons)

cc_wak = compute_CrossCorrs(spikes, wake_ep, norm=True)
cc_slp = compute_CrossCorrs(spikes, sleep_ep, 0.25, 200, norm=True)
cc_wak = cc_wak.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)
cc_slp = cc_slp.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)

s = data_directory.split("/")[-1]

new_index = [(s+'_'+str(i),s+'_'+str(j)) for i,j in cc_wak.columns]
cc_wak.columns = pd.Index(new_index)
cc_slp.columns = pd.Index(new_index)

pairs = pd.Series(index = new_index)
for i,j in pairs.index:	
	if i in neurons and j in neurons:
		a = peaks[i] - peaks[j]
		pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))

pairs = pairs.dropna().sort_values()


# PLOT
from matplotlib import gridspec
titles = ['wake', 'sleep']
fig = figure()
outergs = gridspec.GridSpec(2,2, figure = fig)
gsA = gridspec.GridSpecFromSubplotSpec(2,15, outergs[0,:])
for i, n in enumerate(tcurves.columns[0:15]):
	subplot(gsA[0,i], projection = 'polar')
	plot(tcurves[n])
for i, n in enumerate(tcurves.columns[15:]):
	subplot(gsA[1,i], projection = 'polar')
	plot(tcurves[n])


for i, cc in enumerate([cc_wak, cc_slp]):
	subplot(outergs[1,i])
	imshow(cc[pairs.index].T, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
	title(titles[i])
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])



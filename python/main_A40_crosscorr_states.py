import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean

# data_directory = '/mnt/DataGuillaume/LMN/A1410/A1410-200116A/A1410-200116A'
data_directory = '/mnt/LocalHDD/A4002/A4002-200121_0'

episodes = ['sleep', 'wake']
# events = ['1', '3']
events = ['1']

s = 'A4002-200121_0'

spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					

# acceleration						= loadAuxiliary(data_directory, 6)
# sleep_ep							= refineSleepFromAccel(acceleration, sleep_ep)

sleep_ep	= nts.IntervalSet(start = sleep_ep.loc[0,'end']/2, end = sleep_ep.loc[0,'end'])


tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tcurves 							= smoothAngularTuningCurves(tuning_curves, 10, 2)
tokeep, stat 						= findHDCells(tcurves)
tcurves 							= tuning_curves[tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]

neurons 							= [s+'_'+str(n) for n in tcurves.columns.values]

peaks.index							= pd.Index(neurons)
tcurves.columns						= pd.Index(neurons)

cc_wak = compute_CrossCorrs(spikes, wake_ep, norm=True)
cc_slp = compute_CrossCorrs(spikes, sleep_ep, 0.25, 200, norm=True)
cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
cc_slp = cc_slp.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)

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

titles = ['wake', 'sleep']
figure()
for i, cc in enumerate([cc_wak, cc_slp]):
	subplot(1,2,i+1)
	imshow(cc[pairs.index].T, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
	title(titles[i])
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])



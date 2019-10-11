import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

# data_directory = '../data/A1400/A1407/'
data_directory = '/mnt/DataGuillaume/LMN/A1407/'

info = pd.read_csv(data_directory+'A1407.csv')
info = info.set_index('Session')

# path 								= '../data/A1400/A1407/A1407-190416'
path 								= '/mnt/DataGuillaume/LMN/A1407/A1407-190416'
spikes, shank 						= loadSpikeData(path)
n_channels, fs, shank_to_channel 	= loadXML(path)
episodes 							= info.filter(like='Trial').loc[path.split("/")[-1]].dropna().values
events								= list(np.where(episodes == 'wake')[0].astype('str'))
position 							= loadPosition(path, events, episodes)
wake_ep 							= loadEpoch(path, 'wake', episodes)
sleep_ep 							= loadEpoch(path, 'sleep')					



tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tokeep, stat 						= findHDCells(tuning_curves[1])

for n in tuning_curves.keys():
	tuning_curves[n]				= smoothAngularTuningCurves(tuning_curves[n])

# order tuning curves by peak
order 								= tuning_curves[1].idxmax().argsort().values
for n in tuning_curves.keys():
	tuning_curves[n] 					= tuning_curves[n][tuning_curves[n].columns[order]]


# center tuning curves
centered							= {i:[] for i in tuning_curves.keys()}
for n in tuning_curves[1]:
	peak = tuning_curves[1][n].idxmax()
	newidx = tuning_curves[1][n].index.values - peak
	newidx %= 2*np.pi
	newidx[newidx>np.pi] -= 2*np.pi
	idx = np.argsort(newidx)
	for i in tuning_curves.keys():
		centered[i].append(tuning_curves[i][n].iloc[idx].values)
idx = np.arange(0, 2*np.pi, 2*np.pi/tuning_curves[0].shape[0])
idx[idx>np.pi] -= 2*np.pi
idx = np.sort(idx)
for i in centered.keys():
	tmp = np.array(centered[i]).T
	tmp -= tmp.min(0)
	tmp /= tmp.max(0)
	centered[i] = pd.DataFrame(index = idx, data = tmp, columns = tuning_curves[i].columns)


colors = ['blue', 'black', 'red']

figure()
for i,n in enumerate(tuning_curves[1].columns):
	subplot(3,5,i+1)
	for j in range(3):
		plot(tuning_curves[j][n], color = colors[j])
 


figure()
for i,n in enumerate(centered[1].columns):
	subplot(3,5,i+1)
	for j in range(3):
		plot(centered[j][n], color = colors[j])

figure()

plot(centered[0], color = colors[0])
plot(centered[2], color = colors[2])

show()








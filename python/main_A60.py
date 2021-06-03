import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean


data_directory = '/mnt/Data2/LMN-ADN-PSB/A6004/A6004-210506A'


episodes = ['sleep', 'wake']
# episodes = ['sleep', 'wake', 'sleep', 'wake', 'wake', 'wake', 'sleep']
# episodes = ['wake', 'sleep']
events = ['1']
# events = ['1', '3', '4', '5']
# events = ['1', '3', '5']






spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
acceleration 						= acceleration[[0,1,2]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)



tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)

tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)

rip_ep, rip_tsd 					= loadRipples(data_directory)

tokeep, stat 						= findHDCells(tuning_curves, z=10, p = 0.001)


############################################################################################### 
# PLOT
###############################################################################################
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue']

shank = shank.flatten()

figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		if i in tokeep:
			plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1], linewidth = 3)
		legend()
		count+=1
		gca().set_xticklabels([])


cc_rip = compute_EventCrossCorr(spikes, rip_tsd, sleep_ep, binsize = 10, nbins = 400, norm=True)
cc_rip = cc_rip.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2)


for i, s in enumerate(np.unique(shank)):
	figure()
	for j, n in enumerate(np.where(shank==s)[0]):
		subplot(int(np.sqrt(np.sum(shank==s)))+1,int(np.sqrt(np.sum(shank==s)))+1,j+1)
		plot(cc_rip.iloc[:,n])



				

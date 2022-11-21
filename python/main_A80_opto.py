import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec


data_directory = '/media/sofia/Electrophysiology/A8034/A8034-221013'


episodes = ['sleep', 'sleep', 'wake']
# events = ['1', '3']

# episodes = ['sleep', 'wake', 'sleep']
events = ['2']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)




position 							= loadPosition(data_directory, events, episodes, 2, 1)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory)
#sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

#################
# TUNING CURVES
tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60)
#tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)


tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)

tokeep, stat 						= findHDCells(tuning_curves, z=10, p = 0.001)

#tcurves 							= tuning_curves[1][tokeep]
#peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
#tcurves 							= tcurves[peaks.index.values]




#################
# OPTO
opto_ep = loadOptoEp(data_directory, epoch=1, n_channels = 2, channel = 0)


################
# SLEEP
################
opto_ep = opto_ep.merge_close_intervals(40000)
frates, rasters, bins, stim_duration = computeRasterOpto(spikes, opto_ep, 50)



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue']
shank = shank.flatten()


figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		# plot(tuning_curves2[1][i], '--', color = colors[shank[i]-1])
		if i in tokeep:
			plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1], linewidth = 3)
		legend()
		count+=1
		gca().set_xticklabels([])



groups = np.array_split(list(spikes.keys()), 3)
stim_duration = opto_ep.loc[0,'end'] - opto_ep.loc[0,'start']
for i, neurons in enumerate(groups):		
	# i = 0
	# neurons = groups[0]
	figure()
	count = 1
	for k,n in enumerate(neurons):
		ax = subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,count)
		subgs = GridSpecFromSubplotSpec(2, 2, ax)
		subplot(subgs[:,0], projection = 'polar')
		plot(tuning_curves[n], label = str(shank[n]) + ' ' + str(n), color = colors[shank[n]-1])		
		xticks([])
		yticks([])	
		subplot(subgs[0,1])		
		bar(frates[n].index.values, frates[n].values, np.diff(frates[n].index.values)[0])
		axvline(stim_duration)
		axvline(stim_duration*2)
		title(n+2)
		yticks([])
		subplot(subgs[1,1])
		plot(rasters[n], '.', markersize = 0.24)
		title(n+2)
		count+=1
		gca().set_xticklabels([])
		axvline(stim_duration)
		axvline(stim_duration*2)
		yticks([])

#tight_layout()
show()

sys.exit()


############ WAKE##################################
# WAAKE
################
opto_ep = loadOptoEp(data_directory, epoch=3, n_channels = 2, channel = 0)
opto_ep = opto_ep.merge_close_intervals(40000)
frates, rasters, bins, stim_duration = computeRasterOpto(spikes, opto_ep, 1000)
stim_duration = opto_ep.loc[0,'end'] - opto_ep.loc[0,'start']
start = []
end = []
for i in opto_ep.index.values:
	start.append(opto_ep.loc[i,'start']-stim_duration)
	end.append(opto_ep.loc[i,'start'])

nonopto_ep = nts.IntervalSet(start = start, end = end)

tc_nopto = computeAngularTuningCurves(spikes, position['ry'], nonopto_ep, 60)
tc_nopto = smoothAngularTuningCurves(tc_nopto, 10, 2)
tc_opto = computeAngularTuningCurves(spikes, position['ry'], opto_ep, 60)
tc_opto = smoothAngularTuningCurves(tc_opto, 10, 2)





groups = np.array_split(list(spikes.keys()), 1)
for i, neurons in enumerate(groups):		
	figure()
	count = 1
	for k,n in enumerate(neurons):
		ax = subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,count)
		subgs = GridSpecFromSubplotSpec(2, 2, ax)
		subplot(subgs[:,0], projection = 'polar')
		plot(tc_opto[n], '--', label = str(shank[n]) + ' ' + str(n), color = colors[shank[n]-1])		
		plot(tc_nopto[n], '-', label = str(shank[n]) + ' ' + str(n), color = colors[shank[n]-1])		
		subplot(subgs[0,1])		
		bar(frates[n].index.values, frates[n].values, np.diff(frates[n].index.values)[0])
		axvline(stim_duration)
		axvline(stim_duration*2)
		xlim(frates.index[0], frates.index[-1])
		title(n)
		subplot(subgs[1,1])
		plot(rasters[n], '.', markersize = 0.24)
		count+=1
		gca().set_xticklabels([])
		xlim(frates.index[0], frates.index[-1])
		axvline(stim_duration)
		axvline(stim_duration*2)



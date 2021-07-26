import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec



data_directory = '/mnt/Data2/Opto/A8000/A8008/A8008-210603'

episodes = ['sleep', 'wake', 'sleep']
# events = ['1', '3']

# episodes = ['sleep', 'wake', 'sleep']
events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)




position 							= loadPosition(data_directory, events, episodes, 2, 0)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory)
#sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

#################
# TUNING CURVES
tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
#tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)


tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)

tokeep, stat 						= findHDCells(tuning_curves, z=10, p = 0.001)

#tcurves 							= tuning_curves[1][tokeep]
#peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
#tcurves 							= tcurves[peaks.index.values]

spks = shuffleByIntervalSpikes(spikes, wake_ep)

sys.exit()

#################
# OPTO
opto_ep = loadOptoEp(data_directory, epoch=2, n_channels = 1, channel = 0)


################
# SLEEP
################
opto_ep = opto_ep.merge_close_intervals(40000)
frates, rasters, bins, stim_duration = computeRasterOpto(spikes, opto_ep, 1000)



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
		# legend()
		count+=1
		gca().set_xticklabels([])



groups = np.array_split(list(spikes.keys()), 2)
for i, neurons in enumerate(groups):		
	figure()
	count = 1
	for k,n in enumerate(neurons):
		ax = subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,count)
		subgs = GridSpecFromSubplotSpec(2, 2, ax)
		subplot(subgs[:,0], projection = 'polar')
		plot(tuning_curves[n], label = str(shank[n]) + ' ' + str(n), color = colors[shank[n]-1])		
		subplot(subgs[0,1])		
		bar(frates[n].index.values, frates[n].values, np.diff(frates[n].index.values)[0])
		#axvline(20000000)
		#axvline(40000000)
		title(n)
		subplot(subgs[1,1])
		plot(rasters[n], '.', markersize = 0.24)
		count+=1
		gca().set_xticklabels([])
		#axvline(20000000)
		#axvline(40000000)


sys.exit()

############ WAKE##################################
# WAAKE
################
opto_ep = loadOptoEp(data_directory, epoch=1, n_channels = 2, channel = 1)
opto_ep = opto_ep.merge_close_intervals(500000)
frates, rasters, bins, stim_duration = computeRasterOpto(spikes, opto_ep, 100)

rasters = {}
frates = {}

# assuming all opto stim are the same for a session
#stim_duration = opto_ep.loc[0,'end'] - opto_ep.loc[0,'start']
stim_duration = 10000
bin_size = 2
bins = np.arange(0, stim_duration + 2*stim_duration + bin_size*1000, bin_size*1000)

for n in spikes.keys():
	rasters[n] = []
	r = []
	for e in opto_ep.index:
		ep = nts.IntervalSet(start = opto_ep.loc[e,'start'] - stim_duration,
							end = opto_ep.loc[e,'end'] + stim_duration)
		spk = spikes[n].restrict(ep)
		tmp = pd.Series(index = spk.index.values - ep.loc[0,'start'], data = e)
		rasters[n].append(tmp)
		count, _ = np.histogram(tmp.index.values, bins)
		r.append(count)
	r = np.array(r)
	frates[n] = pd.Series(index = bins[0:-1]/1000, data = r.mean(0))
	rasters[n] = pd.concat(rasters[n])		

frates = pd.concat(frates, 1)
frates = nts.TsdFrame(t = frates.index.values, d = frates.values, time_units = 'ms')

writeNeuroscopeEvents(os.path.join(data_directory, os.path.basename(data_directory)+'.py.opt.evt'), opto_ep, 'opto')


groups = np.array_split(list(spikes.keys()), 1)
for i, neurons in enumerate(groups):		
	figure()
	count = 1
	for k,n in enumerate(neurons):
		ax = subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,count)
		subgs = GridSpecFromSubplotSpec(2, 2, ax)
		subplot(subgs[:,0], projection = 'polar')
		plot(tuning_curves[n], label = str(shank[n]) + ' ' + str(n), color = colors[shank[n]-1])		
		subplot(subgs[0,1])		
		bar(frates[n].index.values, frates[n].values, np.diff(frates[n].index.values)[0])
		axvline(10000)
		axvline(20000)
		xlim(frates.index[0], frates.index[-1])
		title(n)
		subplot(subgs[1,1])
		plot(rasters[n], '.', markersize = 0.24)
		count+=1
		gca().set_xticklabels([])
		axvline(10000)
		axvline(20000)
		xlim(frates.index[0], frates.index[-1])


meanwavef, maxch = loadMeanWaveforms(data_directory)


figure()
subplot(121)
for i in range(53):
	plot(data[label==0].mean(0)[:,i]+100*i)
subplot(122)
for i in range(53):
	plot(data[label==1].mean(0)[:,i]+100*i)



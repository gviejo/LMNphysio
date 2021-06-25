import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec



data_directory = '/mnt/Data2/Opto/A8000/A8009/A8009-210609A'

episodes = ['sleep', 'sleep']
# events = ['1', '3']

# episodes = ['sleep', 'wake', 'sleep']
events = []


#spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
sleep_ep 							= loadEpoch(data_directory, 'sleep')

rip_ep, rip_tsd 					= loadRipples(data_directory)

opto_ep = loadOptoEp(data_directory, epoch=1, n_channels = 1, channel = 0)
opto_ep2 = opto_ep.merge_close_intervals(30e6)

opto_ep2 = opto_ep2.loc[0:14]

rip_spikes = rip_tsd.restrict(sleep_ep.loc[[1]])

rasters = {}
frates = {}

tmp = opto_ep.intersect(opto_ep2.loc[[0]]).merge_close_intervals(40000)
bins = tmp.values.flatten()
bins = bins - bins[0]

bins2 = np.unique(np.hstack(((bins-bins[-2])[:-1],bins,(bins+bins[-1])[2:])))


rasters = []
rate = []
for e in opto_ep2.index:
	start = opto_ep2.loc[e,'start']	
	idx = np.digitize(rip_spikes.index.values, bins2+start)
	idx = idx[~np.logical_or(idx==0, idx==len(bins2))]
	r = np.zeros(len(bins2)-1)
	for j in idx: r[j-1] += 1
	rate.append(r)

	ep = nts.IntervalSet(start = start + bins2[0], end = start+(bins2[-1] - bins2[0])/2)

	tmp = rip_spikes.restrict(ep).index.values
	tmp = pd.Series(index = tmp - ep.loc[0,'start'] + bins2[0], data = e)
	rasters.append(tmp)


rate = np.array(rate)

rate = pd.Series(index = bins2[0:-1], data = rate.sum(0))

rasters = pd.concat(rasters)


figure()
subplot(211)		
bar(rate.index.values, rate.values, np.diff(bins2)/2)
# [axvline(t, linewidth=0.1) for t in bins3]
subplot(212)
plot(rasters, '+', markersize = 2)
#gca().set_xticklabels([])
xlim(bins2[0], bins2[-1])
#axvline(stim_duration)
#axvline(stim_duration*2)
bins3 = bins2[np.logical_and(bins2>=0, bins2<bins[-1])]
[axvline(t, linewidth=0.1) for t in bins3]



idx0 = np.where(np.diff(bins2) <= 600000)[0]
idx1 = np.where(np.diff(bins2) >= 600000)[0]

rate0 = rate.iloc[idx0]
rate1 = rate.iloc[idx1]

bar(rate0.index.values, rate0.values, np.diff(bins2)[idx0]/2, color = 'green')
bar(rate1.index.values, rate1.values, np.diff(bins2)[idx1]/2, color = 'red')


sys.exit()
################
# SLEEP
################





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




############ WAKE##################################
# WAAKE
################
opto_ep = loadOptoEp(data_directory, epoch=1, n_channels = 2, channel = 1)
#opto_ep = opto_ep.merge_close_intervals(500000)
frates, rasters = computeRasterOpto(spikes, opto_ep, 1)

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



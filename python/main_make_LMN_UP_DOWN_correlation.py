import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from itertools import combinations

def zscore_rate(rate):
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	return rate


############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_KS25.txt'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

datasets = [s for s in datasets if 'A5002' in s or 'A5011' in s]

allr = []
allrtemp = []
allrcc = []

for s in datasets:
	print(s)
	name = s.split('/')[-1]
	path = os.path.join(data_directory, s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	episodes 							= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	episodes[episodes != 'sleep'] 		= 'wake'
	events								= list(np.where(episodes != 'sleep')[0].astype('str'))	
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sleep_ep 							= loadEpoch(path, 'sleep')					
	sws_ep								= loadEpoch(path, 'sws')
	rem_ep								= loadEpoch(path, 'rem')
	down_ep, up_ep 						= loadUpDown(path)

	# Only taking the first wake ep
	wake_ep = wake_ep.loc[[0]]

	# # Taking only neurons from LMN
	if 'A5002' in s:
		spikes = {n:spikes[n] for n in np.where(shank==3)[0]}

	if 'A5011' in s:
		spikes = {n:spikes[n] for n in np.where(shank==5)[0]}

	speed = computeSpeed(position[['x', 'z']], wake_ep)
	speed = speed.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)
	idx = np.diff((speed > 0.005)*1.0)
	start = np.where(idx == 1)[0]
	end = np.where(idx == -1)[0]
	if start[0] > end[0]:
		start = np.hstack(([0], start))
	if start[-1] > end[-1]:
		end = np.hstack((end, [len(idx)]))

	newwake_ep = nts.IntervalSet(start = speed.index.values[start], end = speed.index.values[end])
	newwake_ep = newwake_ep.drop_short_intervals(1, time_units='s')

	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################
	tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)	
	tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

	# CHECKING HALF EPOCHS
	wake2_ep = splitWake(wake_ep)
	tokeep2 = []
	stats2 = []
	tcurves2 = []
	for i in range(2):
		# tcurves_half = computeLMNAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])[0][1]
		tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
		tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)
		tokeep, stat = findHDCells(tcurves_half)
		tokeep2.append(tokeep)
		stats2.append(stat)
		tcurves2.append(tcurves_half)

	tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
	tokeep2 = np.union1d(tokeep2[0], tokeep2[1])

	# Checking firing rate
	spikes = {n:spikes[n] for n in tokeep}
	mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])	
	# tokeep = mean_frate[(mean_frate.loc[tokeep]>4).all(1)].index.values
	tokeep = mean_frate[mean_frate.loc[tokeep,'sws']>1].index.values

	spikes = {n:spikes[n] for n in tokeep}
	
	# TAKING UP_EP AND DOWN_EP LARGER THAN 100 ms
	up_ep = up_ep.drop_short_intervals(200, time_units = 'ms')
	up_ep = up_ep.drop_long_intervals(2000, time_units = 'ms')
	down_ep = down_ep.drop_short_intervals(100, time_units = 'ms')

	############################################################################################### 
	# PEARSON CORRELATION GLOBAL
	###############################################################################################
	wak_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, newwake_ep, 300, 1))	
	up_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, up_ep, 30, 1))
	down_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, down_ep, 30, 1))
	
	r_wak = np.corrcoef(wak_rate.T)[np.triu_indices(len(tokeep),1)]	
	r_up = np.corrcoef(up_rate.T)[np.triu_indices(len(tokeep),1)]
	r_down = np.corrcoef(down_rate.T)[np.triu_indices(len(tokeep),1)]

	r = pd.DataFrame(data = np.vstack((r_wak, r_up, r_down)).T)

	neurons = [name+'_'+str(n) for n in tokeep]

	pairs = list(combinations(neurons, 2))

	r.index = pd.Index(pairs)
	r.columns = pd.Index(['wak', 'up', 'down'])

	############################################################################################### 
	# PEARSON CORRELATION TEMPORAL
	###############################################################################################
	nb_bins = 11
	sws_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, sws_ep, 20, 1)
	sws_index = nts.Ts(sws_rate.index.values)
	sws_rate = zscore_rate(sws_rate)

	up_idx = sws_index.restrict(up_ep).index.values
	tmp = np.vstack(up_idx) - up_ep['start'].values
	tmp = tmp.astype(np.float32).T
	tmp[tmp<0] = np.nan
	start_to_idx = np.nanmin(tmp, 0)
	tmp = np.vstack(up_ep['end'].values) - up_idx
	tmp = tmp.astype(np.float32)
	tmp[tmp<0] = np.nan
	idx_to_end = np.nanmin(tmp, 0)
	d_up = start_to_idx/(start_to_idx + idx_to_end)
	sws_index.loc[up_idx] = np.digitize(d_up, np.linspace(0, 1, nb_bins))-1

	dw_idx = sws_index.restrict(down_ep).index.values
	tmp = np.vstack(dw_idx) - down_ep['start'].values
	tmp = tmp.astype(np.float32).T
	tmp[tmp<0] = np.nan
	start_to_idx = np.nanmin(tmp, 0)
	tmp = np.vstack(down_ep['end'].values) - dw_idx
	tmp = tmp.astype(np.float32)
	tmp[tmp<0] = np.nan
	idx_to_end = np.nanmin(tmp, 0)
	d_dw = start_to_idx/(start_to_idx + idx_to_end)	
	sws_index.loc[dw_idx] = np.digitize(d_dw, np.linspace(0, 1, nb_bins))-nb_bins

	sws_rate = sws_rate[~sws_index.as_series().isna()]
	sws_index = sws_index.as_series().dropna().astype(np.int)

	r_temp = pd.DataFrame(index = pairs, columns = np.unique(sws_index.values), dtype = np.float32)
	
	for i in r_temp.columns:
		rate_i = sws_rate[sws_index==i]
		r_i = np.corrcoef(rate_i.T)[np.triu_indices(len(tokeep),1)]
		r_temp[i] = r_i

	############################################################################################### 
	# PEARSON CORRELATION CROSS-CORR WINDOWs UP START
	###############################################################################################
	bins = np.arange(-400, 440, 30)
	
	sws_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, sws_ep, 20, 1)
	sws_index = pd.Series(index = sws_rate.index.values,data = np.nan)
	sws_rate = zscore_rate(sws_rate)

	up_start = up_ep['start'].values
	for t in up_start:
		idx = np.digitize(sws_index.index.values, t+(bins*1000))
		ts = sws_index.index.values[np.logical_and(idx>0, idx<len(bins))]
		sws_index.loc[ts] = idx[np.logical_and(idx>0, idx<len(bins))]-1

	sws_rate = sws_rate[~sws_index.isna()]
	sws_index = sws_index.dropna().astype(np.int)

	r_cc = pd.DataFrame(index = pairs, columns = np.unique(sws_index.values), dtype = np.float32)
	
	for i in r_cc.columns:
		rate_i = sws_rate[sws_index==i]
		r_i = np.corrcoef(rate_i.T)[np.triu_indices(len(tokeep),1)]
		r_cc[i] = r_i

	r_cc.columns = bins[0:-1] + np.diff(bins)/2

	#######################
	# SAVING
	#######################
	allr.append(r)
	allrtemp.append(r_temp)
	allrcc.append(r_cc)


allr = pd.concat(allr, 0)
allrtemp = pd.concat(allrtemp, 0)
allrcc = pd.concat(allrcc, 0)

rtemp = pd.Series(index = allrtemp.columns, dtype = np.float32)
for i in allrtemp.columns:
	m, b = np.polyfit(allr['wak'].values, allrtemp[i].values, 1)
	rtemp.loc[i] = m

rcc = pd.Series(index = allrcc.columns, dtype = np.float32)
for i in allrcc.columns:
	m, b = np.polyfit(allr['wak'].values, allrcc[i].values, 1)
	rcc.loc[i] = m



from matplotlib.gridspec import GridSpec

figure()
subplot(121)
plot(allr['wak'], allr['up'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['up'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('up')
title('r = '+str(np.round(m, 3)))

subplot(122)
plot(allr['wak'], allr['down'], 'o', color = 'grey', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['down'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('down')
title('r = '+str(np.round(m, 3)))




figure()
gs = GridSpec(1, 2, width_ratios = [0.7, 0.3])
ax = subplot(gs[0,0])
plot(allr['wak'], allr['up'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['up'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b, color = 'red', label = 'UP')
plot(allr['wak'], allr['down'], 'o', color = 'grey', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['down'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b, color = 'grey', label = 'DOWN')
xlabel('Wake')
ylabel('NREM')
legend()
subplot(gs[0,1])
bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
hist(allr['up'], bins, color= 'red', alpha = 0.5, orientation = 'horizontal', histtype='step', linewidth = 6)
hist(allr['down'], bins, color= 'grey', alpha = 0.5, orientation = 'horizontal', histtype='step', linewidth = 6)







figure()
subplot(121)
axvline(0)
plot(rcc)
subplot(122)
plot(rtemp)
axvline(0)
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


allr = []


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
	tuning_curves = {1:computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)}
	for i in tuning_curves:
		tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)

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
	tokeep = mean_frate[mean_frate.loc[tokeep,'sws']>2].index.values

	############################################################################################### 
	# PEARSON CORRELATION
	###############################################################################################
	wak_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, newwake_ep, 300, 1))
	rem_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, rem_ep, 300, 1))
	sws_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, sws_ep, 60, 1))

	r_wak = np.corrcoef(wak_rate.T)[np.triu_indices(len(tokeep),1)]
	r_rem = np.corrcoef(rem_rate.T)[np.triu_indices(len(tokeep),1)]
	r_sws = np.corrcoef(sws_rate.T)[np.triu_indices(len(tokeep),1)]

	r = pd.DataFrame(data = np.vstack((r_wak, r_rem, r_sws)).T)

	neurons = [name+'_'+str(n) for n in tokeep]

	pairs = list(combinations(neurons, 2))

	r.index = pd.Index(pairs)
	r.columns = pd.Index(['wak', 'rem', 'sws'])

	#######################
	# SAVING
	#######################
	allr.append(r)

allr = pd.concat(allr, 0)

figure()
subplot(121)
plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('sws')
title('r = '+str(np.round(m, 3)))

subplot(122)
plot(allr['wak'], allr['rem'], 'o', color = 'grey', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['rem'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('rem')
title('r = '+str(np.round(m, 3)))

figure()
plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('sws')
title('r = '+str(np.round(m, 3)))


datatosave = {'allr':allr}
cPickle.dump(datatosave, open(os.path.join('../figures/figures_II_2021', 'All_correlation.pickle'), 'wb'))
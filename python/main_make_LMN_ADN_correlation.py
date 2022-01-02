#!/usr/bin/env python
'''

'''
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv
from matplotlib.gridspec import GridSpec

def zscore_rate(rate):
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	return rate


############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
shanks = pd.read_csv(os.path.join(data_directory,'ADN_LMN_shanks.txt'), header = None, index_col = 0, names = ['ADN', 'LMN'], dtype = np.str)

infos = getAllInfos(data_directory, datasets)


allr = []

for s in datasets:
	print(s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	name 								= s.split('/')[-1]
	path 								= os.path.join(data_directory, s)
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
		tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
		tokeep, stat = findHDCells(tcurves_half)
		tokeep2.append(tokeep)
		stats2.append(stat)
		tcurves2.append(tcurves_half)

	tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

	# NEURONS FROM ADN	
	adn = np.intersect1d(tokeep, np.hstack([np.where(shank == i)[0] for i in np.fromstring(shanks.loc[s,'ADN'], dtype=int,sep=' ')]))
	lmn = np.intersect1d(tokeep, np.hstack([np.where(shank == i)[0] for i in np.fromstring(shanks.loc[s,'LMN'], dtype=int,sep=' ')]))

	tokeep 	= np.hstack((adn, lmn))
	spikes 	= {n:spikes[n] for n in tokeep}

	tcurves 		= tuning_curves[tokeep]
	peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

	speed = computeSpeed(position[['x', 'z']], wake_ep)
	speed = speed.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)
	idx = np.diff((speed > 0.003)*1.0)
	start = np.where(idx == 1)[0]
	end = np.where(idx == -1)[0]
	if len(start):
		if start[0] > end[0]:
			start = np.hstack(([0], start))
		if start[-1] > end[-1]:
			end = np.hstack((end, [len(idx)]))
		newwake_ep = nts.IntervalSet(start = speed.index.values[start], end = speed.index.values[end])
		newwake_ep = newwake_ep.drop_short_intervals(1, time_units='s')
	else:
		newwake_ep = wake_ep

	############################################################################################### 
	# PEARSON CORRELATION
	###############################################################################################
	wak_rate = zscore_rate(binSpikeTrain(spikes, newwake_ep, 100, 2))
	rem_rate = zscore_rate(binSpikeTrain(spikes, rem_ep, 100, 2))	
	sws_rate = zscore_rate(binSpikeTrain(spikes, sws_ep, 20, 5))

	rates = {'wak':pd.DataFrame(wak_rate, columns = tokeep),
		'rem':pd.DataFrame(rem_rate, columns = tokeep),
		'sws':pd.DataFrame(sws_rate, columns = tokeep)}

	from itertools import product
	pairs = list(product(adn, lmn))
	r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)
	for i, j in pairs:
		for ep in rates.keys():
			r.loc[(i,j), ep] = scipy.stats.pearsonr(rates[ep][i],rates[ep][j])[0]

	pairs = list(product([name+'_'+str(n) for n in adn], [name+'_'+str(n) for n in lmn]))

	r.index = pd.Index(pairs)
	
	#######################
	# SAVING
	#######################
	allr.append(r)

allr = pd.concat(allr, 0)


datatosave = {'allr':allr}
cPickle.dump(datatosave, open(os.path.join('../data/', 'All_correlation_ADN_LMN.pickle'), 'wb'))


figure()
subplot(131)
plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('sws')
r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
title('r = '+str(np.round(r, 3)))

subplot(132)
plot(allr['wak'], allr['rem'], 'o',  alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['rem'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('up')
r, p = scipy.stats.pearsonr(allr['wak'], allr['rem'])
title('r = '+str(np.round(r, 3)))

show()


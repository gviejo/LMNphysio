#!/usr/bin/env python
'''

'''
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec




############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)



allr = []

for s in datasets:
	print(s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	path = os.path.join(data_directory, s)
	data = nap.load_session(path, 'neurosuite')
	spikes = data.spikes
	position = data.position
	wake_ep = data.epochs['wake']
	sws_ep = data.read_neuroscope_intervals('sws')
	rem_ep = data.read_neuroscope_intervals('rem')
	idx = spikes._metadata[spikes._metadata["location"].str.contains("adn|lmn")].index.values
	spikes = spikes[idx]
	
	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################
	tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
	tuning_curves = smoothAngularTuningCurves(tuning_curves)
	
	# CHECKING HALF EPOCHS
	wake2_ep = splitWake(position.time_support.loc[[0]])	
	tokeep2 = []
	stats2 = []
	tcurves2 = []	
	for i in range(2):
		tcurves_half = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
		tcurves_half = smoothAngularTuningCurves(tcurves_half)

		tokeep, stat = findHDCells(tcurves_half)
		tokeep2.append(tokeep)
		stats2.append(stat)
		tcurves2.append(tcurves_half)		
	tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
	

	spikes = spikes[tokeep]
	groups = spikes._metadata.loc[tokeep].groupby("location").groups

	tcurves 		= tuning_curves[tokeep]
	peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

	velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
	newwake_ep = velocity.threshold(0.001).time_support	
	
	############################################################################################### 
	# PEARSON CORRELATION
	###############################################################################################
	rates = {}
	for e, ep, bin_size, std in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], [0.1, 0.1, 0.02], [2, 2, 5]):
		count = spikes.count(bin_size, ep)
		rate = count/bin_size
		rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
		rate = zscore_rate(rate)
		rates[e] = rate	

	from itertools import product
	pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
	pairs = pd.MultiIndex.from_tuples(pairs, names=['first', 'second'])
	r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)
	for p in r.index:
		for ep in rates.keys():
			r.loc[p, ep] = scipy.stats.pearsonr(rates[ep][int(p[0])],rates[ep][int(p[1])])[0]

	name = data.basename
	pairs = list(product([name+'_'+str(n) for n in groups['adn']], [name+'_'+str(n) for n in groups['lmn']]))
	pairs = pd.MultiIndex.from_tuples(pairs)
	r.index = pairs
	
	#######################
	# SAVING
	#######################
	allr.append(r)

allr = pd.concat(allr, 0)


# datatosave = {'allr':allr}
# cPickle.dump(datatosave, open(os.path.join('../data/', 'All_correlation_ADN_LMN.pickle'), 'wb'))


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
ylabel('rem')
r, p = scipy.stats.pearsonr(allr['wak'], allr['rem'])
title('r = '+str(np.round(r, 3)))

show()


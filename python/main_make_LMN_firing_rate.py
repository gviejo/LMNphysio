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
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
shanks = pd.read_csv(os.path.join(data_directory,'LMN_shanks.txt'), header = None, index_col = 0, names = ['LMN'], dtype = np.str)

infos = getAllInfos(data_directory, datasets)


allfr = []
count = []

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

	# selecting lmn neurons
	neurons = np.where(np.sum([shank==i for i in np.fromstring(shanks.iloc[1].values[0], dtype=int, sep=' ')], 0))[0]

	tuning_curves = computeAngularTuningCurves({n:spikes[n] for n in neurons}, position['ry'], wake_ep, 121)	
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
	# Firing rate
	###############################################################################################
	frate = computeMeanFiringRate({n:spikes[n] for n in tokeep}, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])

	frate.index = [name+'_'+str(n) for n in tokeep]

	allfr.append(frate)

allfr = pd.concat(allfr, 0)


figure()

subplot(1,2,1)
plot(allfr['wake'], allfr['rem'], 'o', color = 'red')
scipy.stats.pearsonr(allfr['wake'], allfr['rem'])
xlabel('Wake')
ylabel('REM')
subplot(1,2,2)
plot(np.log(allfr['wake']), np.log(allfr['sws']), 'o', color = 'red')
xlabel('Wake')
ylabel('SWS')
all
scipy.stats.pearsonr(np.log(allfr['wake'].values.astype(np.float)), np.log(allfr['sws'].values.astype(np.float)))

show()
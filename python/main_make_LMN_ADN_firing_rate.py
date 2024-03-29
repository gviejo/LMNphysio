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

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
shanks = pd.read_csv(os.path.join(data_directory,'ADN_LMN_shanks.txt'), header = None, index_col = 0, names = ['ADN', 'LMN'], dtype = np.str)

infos = getAllInfos(data_directory, datasets)



allfr = []
alladn = []
alllmn = []

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
	#down_ep, up_ep 						= loadUpDown(path)

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
				
	#################################
	#
	##############################
	frate = computeMeanFiringRate(spikes, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])
	
	frate.index = [name+'_'+str(n) for n in spikes.keys()]

	allfr.append(frate)
	alladn.append([name+'_'+str(n) for n in adn])
	alllmn.append([name+'_'+str(n) for n in lmn])

allfr = pd.concat(allfr, 0)

alladn = np.hstack(alladn)
alllmn = np.hstack(alllmn)

figure()

subplot(2,2,1)
plot(allfr.loc[alladn, 'wake'], allfr.loc[alladn, 'rem'], 'o', color = 'red')
xlabel('Wake')
ylabel('REM')
subplot(2,2,2)
plot(allfr.loc[alladn, 'wake'], allfr.loc[alladn, 'sws'], 'o', color = 'red')
xlabel('Wake')
ylabel('SWS')
subplot(2,2,3)
plot(allfr.loc[alllmn, 'wake'], allfr.loc[alllmn, 'rem'], 'o', color = 'green')
xlabel('Wake')
ylabel('REM')
subplot(2,2,4)
plot(allfr.loc[alllmn, 'wake'], allfr.loc[alllmn, 'sws'], 'o', color = 'green')
xlabel('Wake')
ylabel('SWS')

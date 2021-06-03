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


############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_KS25.txt'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

datasets = [s for s in datasets if 'A5002' in s or 'A5011' in s]

allmua = []

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

	# # Taking only neurons from LMN
	if 'A5002' in s:
		spikes = {n:spikes[n] for n in np.where(shank==3)[0]}

	if 'A5011' in s:
		spikes = {n:spikes[n] for n in np.where(shank==5)[0]}
	
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

	# Checking firing rate
	spikes = {n:spikes[n] for n in tokeep}

	# TAKING UP_EP AND DOWN_EP LARGER THAN 100 ms
	up_ep = up_ep.drop_short_intervals(200, time_units = 'ms')
	down_ep = down_ep.drop_short_intervals(200, time_units = 'ms')	

	######################################################################################################
	# HD RATES / UP DOWN	
	######################################################################################################

	mua = []
	bins = np.hstack((np.linspace(0,1,200)-1,np.linspace(0,1,200)[1:]))	
	for n in spikes.keys():
		spk = spikes[n].restrict(up_ep).index.values
		spk2 = np.array_split(spk, 10)

		start_to_spk = []
		for i in range(len(spk2)):
			tmp1 = np.vstack(spk2[i]) - up_ep['start'].values
			tmp1 = tmp1.astype(np.float32).T
			tmp1[tmp1<0] = np.nan
			start_to_spk.append(np.nanmin(tmp1, 0))
		start_to_spk = np.hstack(start_to_spk)

		spk_to_end = []
		for i in range(len(spk2)):
			tmp2 = np.vstack(up_ep['end'].values) - spk2[i]
			tmp2 = tmp2.astype(np.float32)
			tmp2[tmp2<0] = np.nan
			spk_to_end.append(np.nanmin(tmp2, 0))
		spk_to_end = np.hstack(spk_to_end)

		d = start_to_spk/(start_to_spk+spk_to_end)
		mua_up = d

		spk = spikes[n].restrict(down_ep).index.values
		tmp1 = np.vstack(spk) - down_ep['start'].values
		tmp1 = tmp1.astype(np.float32).T
		tmp1[tmp1<0] = np.nan
		start_to_spk = np.nanmin(tmp1, 0)

		tmp2 = np.vstack(down_ep['end'].values) - spk
		tmp2 = tmp2.astype(np.float32)
		tmp2[tmp2<0] = np.nan
		spk_to_end = np.nanmin(tmp2, 0)

		d = start_to_spk/(start_to_spk+spk_to_end)
		mua_down = d

		p, _ = np.histogram(np.hstack((mua_down-1,mua_up)), bins)

		mua.append(p)

	mua = pd.DataFrame(
		index = bins[0:-1]+np.diff(bins)/2, 
		data = np.array(mua).T,
		columns = [name+'_'+str(n) for n in spikes.keys()])

	allmua.append(mua)

allmua = pd.concat(allmua, 1)
allmua = allmua/allmua.sum(0)

figure()
plot(allmua, alpha = 0.5, color = 'grey')
plot(allmua.mean(1))
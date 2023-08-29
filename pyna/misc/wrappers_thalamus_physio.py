# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-02 18:31:25
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-02 18:37:55
import numpy as np
import pandas as pd
import pynapple as nap

def loadShankStructure(generalinfo):
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]-1
		else :
			shankStructure[k[0]] = []
	
	return shankStructure	

def loadSpikeData(path, index):
	# units shoud be the value to convert in s 
	import scipy.io	
	spikedata = scipy.io.loadmat(path)
	shank = spikedata['shank'] - 1
	shankIndex = np.where(shank == index)[0]

	spikes = {}	
	for i in shankIndex:	
		spikes[i] = nap.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2], time_units = 's')

	a = spikes[0].as_units('s').index.values	
	if ((a[-1]-a[0])/60.)/60. > 20. : # VERY BAD		
		spikes = {}	
		for i in shankIndex:	
			spikes[i] = nap.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2]*0.0001, time_units = 's')

	spikes = nap.TsGroup(spikes)

	return spikes, shank

def loadEpoch(path, epoch):
	import scipy.io	
	sampling_freq = 1250	
	behepochs = scipy.io.loadmat(path+'/Analysis/BehavEpochs.mat')

	if epoch == 'wake':
		wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
		return nap.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)

	elif epoch == 'sleep':
		sleep_pre_ep, sleep_post_ep = [], []
		if 'sleepPreEp' in behepochs.keys():
			sleep_pre_ep = behepochs['sleepPreEp'][0][0]
			sleep_pre_ep = np.hstack([sleep_pre_ep[1],sleep_pre_ep[2]])
			sleep_pre_ep_index = behepochs['sleepPreEpIx'][0]
		if 'sleepPostEp' in behepochs.keys():
			sleep_post_ep = behepochs['sleepPostEp'][0][0]
			sleep_post_ep = np.hstack([sleep_post_ep[1],sleep_post_ep[2]])
			sleep_post_ep_index = behepochs['sleepPostEpIx'][0]
		if len(sleep_pre_ep) and len(sleep_post_ep):
			sleep_ep = np.vstack((sleep_pre_ep, sleep_post_ep))
		elif len(sleep_pre_ep):
			sleep_ep = sleep_pre_ep
		elif len(sleep_post_ep):
			sleep_ep = sleep_post_ep						
		return nap.IntervalSet(sleep_ep[:,0], sleep_ep[:,1], time_units = 's')

	elif epoch == 'sws':
		import os
		file1 = path.split("/")[-1]+'.sts.SWS'
		file2 = path.split("/")[-1]+'-states.mat'
		listdir = os.listdir(path)
		if file1 in listdir:
			sws = np.genfromtxt(path+'/'+file1)/float(sampling_freq)
			return nap.IntervalSet.drop_short_intervals(nap.IntervalSet(sws[:,0], sws[:,1], time_units = 's'), 0.0)

		elif file2 in listdir:
			sws = scipy.io.loadmat(path+'/'+file2)['states'][0]
			index = np.logical_or(sws == 2, sws == 3)*1.0
			index = index[1:] - index[0:-1]
			start = np.where(index == 1)[0]+1
			stop = np.where(index == -1)[0]
			return nap.IntervalSet.drop_short_intervals(nap.IntervalSet(start, stop, time_units = 's', expect_fix=True), 0.0)

	elif epoch == 'rem':
		import os
		file1 = path.split("/")[-1]+'.sts.REM'
		file2 = path.split("/")[-1]+'-states.mat'
		listdir = os.listdir(path)	
		if file1 in listdir:
			rem = np.genfromtxt(path+'/'+file1)/float(sampling_freq)
			return nap.IntervalSet(rem[:,0], rem[:,1], time_units = 's').drop_short_intervals(0.0)

		elif file2 in listdir:
			rem = scipy.io.loadmat(path+'/'+file2)['states'][0]
			index = (rem == 5)*1.0
			index = index[1:] - index[0:-1]
			start = np.where(index == 1)[0]+1
			stop = np.where(index == -1)[0]
			return nap.IntervalSet(start, stop, time_units = 's', expect_fix=True).drop_short_intervals(0.0)
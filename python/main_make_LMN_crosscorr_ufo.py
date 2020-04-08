import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys


############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = r'D:\Dropbox (Peyrache Lab)\Peyrache Lab Team Folder\Data\LMN'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)


# for s in sessions:
for s in ['A5000/A5002/A5002-200304A']:
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))
	events			= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)
	sleep_ep		= loadEpoch(path, 'sleep')
	sws_ep 			= loadEpoch(path, 'sws')
	rem_ep 			= loadEpoch(path, 'rem')

	acceleration						= loadAuxiliary(path)
	if 'A5001' in s:
		acceleration = acceleration[[3,4,5]]
		acceleration.columns = range(3)
	elif 'A5002' in s:
		acceleration = acceleration[[0,1,2]]
	newsleep_ep 						= refineSleepFromAccel(acceleration, sleep_ep)

	ufo_ep, ufo_tsd	= loadUFOs(path)


	# TO REMOVE
	half_sleep = nts.IntervalSet(start = sws_ep.start[0], end = sws_ep.start[0] + (sws_ep.end.values[-1] - sws_ep.start[0])/4)
	spikes = {n:spikes[n] for n in np.where(shank==3)[0]}



	cc_ufo = compute_EventCrossCorr(spikes, ufo_tsd, half_sleep, binsize = 1, nbins = 200, norm=False)
	plot(cc_ufo)
	show()


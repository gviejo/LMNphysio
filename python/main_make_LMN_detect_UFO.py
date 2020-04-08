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

	acceleration						= loadAuxiliary(path)
	if 'A5001' in s:
		acceleration = acceleration[[3,4,5]]
		acceleration.columns = range(3)
	elif 'A5002' in s:
		acceleration = acceleration[[0,1,2]]
	newsleep_ep 						= refineSleepFromAccel(acceleration, sleep_ep)

		

	##################################################################################################
	# LOADING LFP
	##################################################################################################	
	lfp = pd.read_hdf(path + '/Analysis/lfp_lmn.h5')
	lfp = lfp[39]
	lfp = nts.Tsd(t = lfp.index.values, d = lfp.values, time_units = 's')

	half_sleep = nts.IntervalSet(start = sws_ep.start[0], end = sws_ep.start[0] + (sws_ep.end.values[-1] - sws_ep.start[0])/4)

	# lfp = lfp.restrict(half_sleep)
	# lfp = downsample(lfp, 1, 2)

	ex_ep = nts.IntervalSet(start = 2348.503, end = 2348.594, time_units = 's')
	startex = int(2348.503*1e6)
	stopex  = int(2348.594*1e6)
	
	##################################################################################################
	# DETECTION UFO
	##################################################################################################
	windowLength = 81
	frequency = 20000
	low_cut = 600
	high_cut = 3000

	nSS_highcut = 1000000
	low_thresFactor = 5 # zscored
	high_thresFactor = 100
	minRipLen = 2 # ms
	maxRipLen = 20 # ms
	minInterRippleInterval = 100 # ms
	limit_peak = 100


	signal			= butter_bandpass_filter(lfp, low_cut, high_cut, frequency, 2)

	signal = nts.Tsd(t = lfp.index.values, d = signal)

	squared_signal = np.square(signal.values)

	window = np.ones(windowLength)/windowLength

	nSS = scipy.signal.filtfilt(window, 1, squared_signal)

	nSS = pd.Series(index = lfp.index.values, data = nSS)
	nSS=nts.Tsd(nSS)


	# Removing point above 100000/
	nSS = nSS.as_series()
	nSS = nSS[nSS<nSS_highcut]
	nSS = (nSS - np.mean(nSS))/np.std(nSS)
	nSS = nts.Tsd(nSS)
	
	figure()
	ax = subplot(211)
	plot(lfp.loc[startex:stopex])
	plot(signal.loc[startex:stopex])
	subplot(212, sharex =  ax)
	plot(nSS.loc[startex:stopex])
	axhline(low_thresFactor)
	show()

	
	######################################################l##################################
	# Round1 : Detecting Ripple Periods by thresholding normalized signal
	nSS = nSS.as_series()
	thresholded = np.where(nSS > low_thresFactor, 1,0)
	start = np.where(np.diff(thresholded) > 0)[0]
	stop = np.where(np.diff(thresholded) < 0)[0]
	if len(stop) == len(start)-1:
		start = start[0:]
	if len(stop)-1 == len(start):
		stop = stop[1:]



	################################################################################################
	# Round 2 : Excluding candidates ripples whose length < minRipLen and greater than Maximum Ripple Length
	if len(start):
		l = (nSS.index.values[stop] - nSS.index.values[start])/1000 # from us to ms
		idx = np.logical_and(l > minRipLen, l < maxRipLen)
	else:	
		print("Detection by threshold failed!")
		sys.exit()

	rip_ep = nts.IntervalSet(start = nSS.index.values[start[idx]], end = nSS.index.values[stop[idx]])
	# rip_ep = rip_ep.intersect(sws_ep)


	# figure()
	# plot(lfp.restrict(sleep_ep))
	# plot(lfp.restrict(rip_ep))
	# show()



	####################################################################################################################
	# Round 3 : Merging ripples if inter-ripple period is too short
	rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval/1000, time_units = 's')



	#####################################################################################################################
	# Round 4: Discard Ripples with a peak power < high_thresFactor and > limit_peak
	rip_max = []
	rip_tsd = []
	for s, e in rip_ep.values:
		tmp = nSS.loc[s:e]
		rip_tsd.append(tmp.idxmax())
		rip_max.append(tmp.max())

	rip_max = np.array(rip_max)
	rip_tsd = np.array(rip_tsd)

	# tokeep = np.logical_and(rip_max > high_thresFactor, rip_max < limit_peak)

	# rip_ep = rip_ep[tokeep].reset_index(drop=True)
	# rip_tsd = nts.Tsd(t = rip_tsd[tokeep], d = rip_max[tokeep])

	rip_tsd = nts.Tsd(t = rip_tsd, d = rip_max)
	###########################################################################################################
	# Writing for neuroscope
	start = rip_ep.as_units('ms')['start'].values
	peaks = rip_tsd.as_units('ms').index.values
	ends = rip_ep.as_units('ms')['end'].values

	datatowrite = np.vstack((start,peaks,ends)).T.flatten()

	n = len(rip_ep)

	texttowrite = np.vstack(((np.repeat(np.array(['UFO start 1']), n)), 
							(np.repeat(np.array(['UFO peak 1']), n)),
							(np.repeat(np.array(['UFO stop 1']), n))
								)).T.flatten()

	#evt_file = data_directory+session+'/'+session.split('/')[1]+'.evt.py.rip'
	evt_file = os.path.join(path, name + '.evt.py.ufo')
	f = open(evt_file, 'w')
	for t, n in zip(datatowrite, texttowrite):
		f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
	f.close()	



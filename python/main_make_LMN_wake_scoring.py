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
# data_directory = r'D:\Dropbox (Peyrache Lab)\Peyrache Lab Team Folder\Data\LMN'
data_directory = r'/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/LMN'
# data_directory = '/mnt/DataGuillaume/LMN-ADN/'
# datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)



# for s in datasets:
for s in ['A5000/A5002/A5002-200303B']:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))
	events			= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)


	##################################################################################################
	# DOWNSAMPLING
	##################################################################################################
	if not os.path.exists(os.path.join(data_directory,s,name+'.eeg')):
		downsampleDatFile(os.path.join(data_directory,s))

		

	##################################################################################################
	# LOADING LFP
	##################################################################################################	
	if 'A5001' in s:
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 84, 1250, 'int16')
	elif 'A5002' in s:
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 94, 1250, 'int16')
	elif 'A1407' in s:
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 1, 1250, 'int16')
	elif 'A4002' in s:
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 49, 1250, 'int16')



	lfp 		= downsample(lfp, 1, 5)


	##################################################################################################
	# DETECTION THETA
	##################################################################################################
	lfp_filt_theta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 5, 15, 1250/5, 2))
	power	 		= nts.Tsd(lfp_filt_theta.index.values, np.abs(lfp_filt_theta.values))

	enveloppe,dummy	= getPeaksandTroughs(power, 5)

	ax = subplot(211)
	plot(lfp.restrict(wake_ep))
	plot(lfp_filt_theta.restrict(wake_ep))
	ax2 = subplot(212, sharex = ax)
	plot(power.restrict(wake_ep))
	plot(enveloppe.restrict(wake_ep), 'o')


	speed = computeSpeed(position[['x', 'z']], wake_ep, 0.4)
	speed2 = speed.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=5)

	index 			= (speed2.values > 0.03)*1
	start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
	end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
	if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
	if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
	tmp 			= np.where(end_cand != start_cand)
	start_cand 		= speed2.index.values[start_cand[tmp]]
	end_cand	 	= speed2.index.values[end_cand[tmp]]
	good_ep			= nts.IntervalSet(start_cand, end_cand)


	index 			= (enveloppe.values > 100)*(enveloppe.values < 2000)*1.0
	start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
	end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
	if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
	if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
	tmp 			= np.where(end_cand != start_cand)
	start_cand 		= enveloppe.index.values[start_cand[tmp]]
	end_cand	 	= enveloppe.index.values[end_cand[tmp]]
	good2_ep		= nts.IntervalSet(start_cand, end_cand)


	theta_wake_ep = good_ep.intersect(good2_ep).merge_close_intervals(30000).drop_short_intervals(4000000)

	
	writeNeuroscopeEvents(os.path.join(data_directory,s,name+'.wake.evt.theta'), theta_wake_ep, "Theta")

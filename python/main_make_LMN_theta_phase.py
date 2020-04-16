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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'))
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_DTN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)




# for s in datasets:
for s in ['A5000/A5002/A5002-200304A']:
# for s in ['A5000/A5002/A5002-200309A']:
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
	# sleep_ep		= loadEpoch(path, 'sleep')
	sws_ep 			= loadEpoch(path, 'sws')
	rem_ep 			= loadEpoch(path, 'rem')
	theta_wake_ep	= loadEpoch(path, 'wake.evt.theta')

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
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 9, 1250, 'int16')
	elif 'A1407' in s:
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 1, 1250, 'int16')
	elif 'A4002' in s:
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 49, 1250, 'int16')

	lfp 		= downsample(lfp, 1, 5)


	##################################################################################################
	# DETECTION THETA
	#################################################################################################
	phase 			= getPhase(lfp, 5, 15, 16, 1250/5.)	

	tmp = nts.Tsd(t = phase.index.values, d = np.cos(phase.values)*2000)

	phase 			= phase.restrict(theta_wake_ep)

	spikes_phase	= {n:phase.realign(spikes[n].restrict(theta_wake_ep), align = 'closest') for n in spikes.keys()}

	# theta_mod 	= {}
	# for n in spikes_phase.keys():			
	# 	ph = spikes_phase[n].restrict(theta_wake_ep)
	# 	mu, kappa, pval = getCircularMean(ph.values)
	# 	theta_mod[e][session.split("/")[1]+"_"+str(neuron)] = np.array([mu, pval, kappa])
	# 	spikes_theta_phase[e][session.split("/")[1]+"_"+str(neuron)] = ph.values


	import _pickle as cPickle
	cPickle.dump(spikes_phase, open(path+'/Analysis/spike_theta_wake.pickle', 'wb'))
	

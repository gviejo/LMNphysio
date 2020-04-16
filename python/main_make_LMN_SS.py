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



# for s in datasets[-2:]:
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
	# wake_ep 		= loadEpoch(path, 'wake', episodes)
	sleep_ep		= loadEpoch(path, 'sleep')
	# sws_ep 			= loadEpoch(path, 'sws')

	#####################################
	# PARAMETERS
	#####################################
	# windowLength = 81
	frequency = 20000
	low_cut = 400
	high_cut = 3000
	nSS_highcut = 50
	low_thresFactor = 6
	high_thresFactor = 100
	minRipLen = 2 # ms
	maxRipLen = 10 # ms
	minInterRippleInterval = 3 # ms
	limit_peak = 100

	if 'A5002' in name:
		datfile = '/mnt/DataGuillaume/LMN-ADN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
	elif 'A1407' in name:
		datfile = '/mnt/DataGuillaume/LMN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
		shank_to_channel = {i+2:shank_to_channel[i] for i in shank_to_channel.keys()}
	elif 'A5001' in name:
		datfile = '/mnt/DataGuillaume/LMN-ADN/'+name.split('-')[0] + '/' + name + '/'+ name +'.dat'
		shank_to_channel = {i-3:shank_to_channel[i] for i in [5, 6, 7, 8]}


	frequency = 20000

	f = open(datfile, 'rb')
	startoffile = f.seek(0, 0)
	endoffile = f.seek(0, 2)
	bytes_size = 2		
	n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
	duration = n_samples/frequency
	f.close()
	fp = np.memmap(datfile, np.int16, 'r', shape = (n_samples, n_channels))
	timestep = (np.arange(0, n_samples)/frequency)*1e6
	timestep = timestep.astype(np.int)
	duree = len(timestep)	

	dummy = pd.Series(index = timestep, data = 0)

	# # #TO REMOVE
	# half_sleep = nts.IntervalSet(start = 2000, end = 3000, time_units = 's')
	# duree = len(dummy.loc[half_sleep.start[0]:half_sleep.end[0]].index.values)
	# dummy = dummy.loc[half_sleep.start[0]:half_sleep.end[0]]

	# sys.exit()


	SS = np.zeros((duree,4))

	for i, sh in enumerate([2,3,4,5]):
		for ch in shank_to_channel[sh]:
			print(sh, ch)
			##################################################################################################
			# LOADING LFP
			##################################################################################################	
			lfp = pd.Series(index = timestep, data = fp[:,ch])
			
			##################################################################################################
			# FILTERING
			##################################################################################################
			signal			= butter_bandpass_filter(lfp, low_cut, high_cut, frequency, 6)
			squared_signal = np.square(signal)
					
			SS[:,i] += squared_signal

			del lfp
			del signal
			del squared_signal

		SS[:,i] /= len(shank_to_channel[sh])


	# SS = pd.DataFrame(index = dummy.index, data = SS)

	for i, sh in enumerate([2,3,4,5]):
		idx = list(range(4))
		idx.remove(i)
		tmp = pd.DataFrame(index = dummy.index, 
			data = (SS[:,i] - SS[:,idx].mean(1))/(SS[:,idx].mean(1)+1))

		tmp.to_hdf(path+'/Analysis/SS_'+str(sh)+'.h5', 'ss')
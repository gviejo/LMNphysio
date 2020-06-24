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
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'))
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_DTN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)

datasets = ['LMN-ADN/A5002/'+s for s in infos['A5002'].index[1:-5]]


for s in datasets:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)
	sleep_ep		= loadEpoch(path, 'sleep')
	acceleration						= loadAuxiliary(path, 2)
	if 'A5001' in s and 3 in acceleration.columns:
		acceleration = acceleration[[3,4,5]]
		acceleration.columns = range(3)
	elif 'A5002' in s:
		acceleration = acceleration[[0,1,2]]
	newsleep_ep 						= refineSleepFromAccel(acceleration, sleep_ep)


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
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 80, 1250, 'int16')
	elif 'A1407' in s:
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 1, 1250, 'int16')
	elif 'A4002' in s:
		lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 49, 1250, 'int16')

	lfp 		= downsample(lfp, 1, 5)


	##################################################################################################
	# DETECTION THETA
	##################################################################################################
	lfp_filt_theta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 4, 12, 1250/5, 2))
	power_theta		= nts.Tsd(lfp_filt_theta.index.values, np.abs(lfp_filt_theta.values))
	# power_theta		= power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=40)
	power_theta		= power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

	lfp_filt_delta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 0.5, 4, 1250/5, 2))
	power_delta		= nts.Tsd(lfp_filt_delta.index.values, np.abs(lfp_filt_delta.values))
	# power_delta		= power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=40)
	power_delta		= power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

	ratio 			= nts.Tsd(t = power_theta.index.values, d = np.log(power_theta.values/power_delta.values))

	ratio2			= ratio.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=200)
	ratio2 			= nts.Tsd(t = ratio2.index.values, d = ratio2.values)


	index 			= (ratio2.as_series() > 0).values*1.0
	start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
	end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
	if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
	if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
	tmp 			= np.where(end_cand != start_cand)
	start_cand 		= ratio2.index.values[start_cand[tmp]]
	end_cand	 	= ratio2.index.values[end_cand[tmp]]
	good_ep			= nts.IntervalSet(start_cand, end_cand)
	good_ep			= newsleep_ep.intersect(good_ep)
	# good_ep			= good_ep.drop_short_intervals(5, time_units = 's')
	good_ep			= good_ep.merge_close_intervals(10, time_units = 's')
	good_ep			= good_ep.drop_short_intervals(20, time_units = 's')
	good_ep			= good_ep.reset_index(drop=True)
	# good_ep			= good_ep.merge_close_intervals(0.5, time_units = 's')
	
	

	theta_rem_ep	= good_ep

	sws_ep 	= newsleep_ep.set_diff(theta_rem_ep)
	sws_ep = sws_ep.merge_close_intervals(0).drop_short_intervals(0)

	# figure()
	# ax = subplot(211)
	# [plot(lfp.restrict(theta_rem_ep.loc[[i]]), color = 'blue') for i in theta_rem_ep.index]
	# [plot(lfp.restrict(sws_ep.loc[[i]]), color = 'orange') for i in sws_ep.index]
	# plot(lfp_filt_theta.restrict(newsleep_ep))
	# subplot(212, sharex = ax)
	# [plot(ratio.restrict(theta_rem_ep.loc[[i]]), color = 'blue') for i in theta_rem_ep.index]
	# [plot(ratio.restrict(sws_ep.loc[[i]]), color = 'orange') for i in sws_ep.index]
	# plot(ratio2.restrict(newsleep_ep))

	# axhline(0)
	# show()

	# sys.exit()

	writeNeuroscopeEvents(os.path.join(data_directory,s,name+'.rem.evt'), theta_rem_ep, "Theta")
	writeNeuroscopeEvents(os.path.join(data_directory,s,name+'.sws.evt'), sws_ep, "SWS")



sys.exit()

phase 			= getPhase(lfp_hpc, 6, 14, 16, fs/5.)	
ep 				= { 'wake'	: theta_wake_ep,
					'rem'	: theta_rem_ep}
theta_mod 		= {}



for e in ep.keys():		
	spikes_phase	= {n:phase.realign(spikes[n], align = 'closest') for n in spikes.keys()}

	# theta_mod[e] 	= np.ones((n_neuron,3))*np.nan
	theta_mod[e] 	= {}
	for n in range(len(spikes_phase.keys())):			
		neuron = list(spikes_phase.keys())[n]
		ph = spikes_phase[neuron].restrict(ep[e])
		mu, kappa, pval = getCircularMean(ph.values)
		theta_mod[e][session.split("/")[1]+"_"+str(neuron)] = np.array([mu, pval, kappa])
		spikes_theta_phase[e][session.split("/")[1]+"_"+str(neuron)] = ph.values


stop = time.time()
print(stop - start, ' s')		
datatosave[session] = theta_mod


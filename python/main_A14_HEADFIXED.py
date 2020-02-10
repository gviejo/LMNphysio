import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

data_directory = '/mnt/DataGuillaume/LMN/A1410/A1410-200112A2'

# episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
episodes = ['wake', 'wake']
events = ['0', '1']

##################################################################################################
# LOADING DATA
##################################################################################################
spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
# sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory)

##################################################################################################
# ANGULAR TUNING CURVES
##################################################################################################
tcurves_fm  						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 121)
tcurves_hf  						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[1]], 121)
tcurves_fm_2, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 121)
# tcurves_hf_2, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[1]], 121)

##################################################################################################
# AUTOCORRS
##################################################################################################
auto_fm, frate_fm 					= compute_AutoCorrs(spikes, wake_ep.loc[[0]])
auto_hf, frate_hf 					= compute_AutoCorrs(spikes, wake_ep.loc[[1]])
# auto_sleep, frate_sleep 			= compute_AutoCorrs(spikes, sleep_ep)

##################################################################################################
# AHV
##################################################################################################
ahv_fm 								= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], nb_bins = 30, norm=False)
ahv_hf 								= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep.loc[[1]], nb_bins = 30, norm=False)

##################################################################################################
# speed
##################################################################################################
speed_fm 							= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep.loc[[0]])
speed_hf 							= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep.loc[[1]])


##################################################################################################
# smoothing
##################################################################################################
for i in tcurves_fm_2:
	tcurves_fm_2[i] = smoothAngularTuningCurves(tcurves_fm_2[i], 10, 2)

tcurves_fm = smoothAngularTuningCurves(tcurves_fm, 10, 2)
tcurves_hf = smoothAngularTuningCurves(tcurves_hf, 10, 2)

ahv_fm = ahv_fm.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
ahv_hf = ahv_hf.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
speed_fm = speed_fm.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
speed_hf = speed_hf.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
		

############################################################################################### 
# PLOT
###############################################################################################


figure()
for i in spikes:
	subplot(6,7,i+1, projection = 'polar')
	plot(tcurves_fm[i], label = str(shank[i]))
	plot(tcurves_hf[i])
	legend()


figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(ahv_fm[i], label = str(shank[i]))
	plot(ahv_hf[i])
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(speed_fm[i], label = str(shank[i]))
	legend()


figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(auto_fm[i], label = str(shank[i]))
	plot(auto_hf[i])
	legend()

show()



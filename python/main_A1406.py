import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

data_directory = '/mnt/DataGuillaume/LMN/A1406/A1406-190311'

# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake']
# episodes = ['wake', 'wake']
episodes = ['wake']
events = ['0']


spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
# sleep_ep 							= loadEpoch(data_directory, 'sleep')					
position 							= loadPosition(data_directory, events)
acceleration						= loadAuxiliary(data_directory)


tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 10)
autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
# autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)
velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep)

# downsampleDatFile(data_directory)

	
figure()
for i in spikes:
	subplot(3,4,i+1, projection = 'polar')
	plot(tuning_curves[i], label = str(shank[i]))
	legend()

figure()
for i in spikes:
	subplot(3,4,i+1)
	plot(autocorr_wake[i], label = str(shank[i]))
	# plot(autocorr_sleep[i])
	legend()

figure()
for i in spikes:
	subplot(3,4,i+1)
	plot(velo_curves[i], label = str(shank[i]))
	legend()

show()



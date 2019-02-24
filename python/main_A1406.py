import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

data_directory = '/mnt/DataGuillaume/LMN/A1406/A1406-190223'

episodes = ['sleep', 'wake', 'sleep']
events = ['1']


if 'Position.h5' not in os.listdir(os.path.join(data_directory, 'Analysis')):
	makeEpochs(data_directory, episodes, file='Epoch_TS.csv')
	makePositions(data_directory, events)

spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
wake_ep 							= loadEpoch(data_directory, 'wake')
sleep_ep 							= loadEpoch(data_directory, 'sleep')
position 							= loadPosition(data_directory)


tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep)
spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 10)
autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)



from pylab import *
figure()
for i in spikes:
	subplot(5,5,i+1)
	plot(tuning_curves[i])


figure()
for i in spikes:
	subplot(5,5,i+1)
	plot(autocorr_wake[i])

figure()
for i in spikes:
	subplot(5,5,i+1)
	imshow(spatial_curves[i], extent = extent)

show()



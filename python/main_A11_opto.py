import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys


data_directory = '/mnt/LocalHDD/ADN/A1110/A1110-180626/A1110-180626'

episodes = ['sleep', 'wake', 'sleep', 'wake']
events = ['1', '3']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)

position.columns = pd.Index(['rx', 'rz', 'ry', 'x', 'y', 'z'])

wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					

acceleration						= loadAuxiliary(data_directory)
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

# LOAD OPTO PULSES



# tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 20)
autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)
velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=False)
# sys.exit()
# mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, sleep_ep], ['wake', 'sleep'])
speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)

# downsampleDatFile(data_directory)

for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)

velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
		

############################################################################################### 
# PLOT
###############################################################################################


figure()
for i in spikes:
	subplot(6,7,i+1, projection = 'polar')
	plot(tuning_curves[1][i], label = str(shank[i]))
	legend()



figure()
subplot(121)
plot(velocity)
subplot(122)
hist(velocity, 1000)
[axvline(e) for e in edges[1:-1]]


figure()
style = ['--', '-', '--']
colors = ['black', 'red', 'black']
alphas = [0.7, 1, 0.7]
for i in spikes:
	subplot(6,7,i+1)
	for j in range(3):
	# for j in [1]:
		tmp = tuning_curves[j][i] #- mean_frate.loc[i,'wake']
		plot(tmp, linestyle = style[j], color = colors[j], alpha = alphas[j])
	title(str(shank[i]))



figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(autocorr_wake[i], label = str(shank[i]))
	# plot(autocorr_sleep[i])
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(velo_curves[i], label = str(shank[i]))
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	imshow(spatial_curves[i], interpolation = 'bilinear')
	colorbar()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(speed_curves[i], label = str(shank[i]))
	legend()

show()



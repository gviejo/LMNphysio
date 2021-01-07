import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

# data_directory = '/mnt/DataGuillaume/LMN/A1407/A1407-190429'
# data_directory = '/mnt/DataGuillaume/LMN/A1407/A1407-190425'
# data_directory = '/mnt/DataGuillaume/PostSub/A3003/A3003-190516A'
#data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200117C/A3004-200117C'
# data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200122B'
# data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200122C'
# data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200124B2'
data_directory = '/mnt/DataGuillaume/LMN-PSB/A3007/A3007-201127A'
episodes = ['sleep', 'wake']
# episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']

events = ['1']





spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					

tuning_curves2 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 120)
tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 20)
autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
# autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)
velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=False)
# sys.exit()
# mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, sleep_ep], ['wake', 'sleep'])
speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)

# downsampleDatFile(data_directory)

tuning_curves2 = smoothAngularTuningCurves(tuning_curves2, 10, 2)
for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)

velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


lfp = loadLFP(os.path.join(data_directory,data_directory.split('/')[-1]+'.dat'), n_channels=64, channel=0, frequency=20000.0, precision='int16')

############################################################################################### 
# PLOT
###############################################################################################

figure()
for i in spikes:
	subplot(4,8,i+1, projection = 'polar')
	plot(tuning_curves2[i], label = str(shank[i])+'-'+str(i))
	legend()




figure()
for i in spikes:
	subplot(4,4,i+1, projection = 'polar')
	plot(tuning_curves[1][i], label = str(shank[i])+'-'+str(i))
	legend()

sys.exit()

figure()
for i,n in enumerate(fs):
	subplot(1,4,i+1, projection = 'polar')
	plot(tuning_curves2[n])
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
	plot(autocorr_wake[i], label = str(shank[i])+'-'+str(i))
	# plot(autocorr_sleep[i])
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(velo_curves[i], label = str(shank[i])+'-'+str(i))
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	imshow(spatial_curves[i], label = str(shank[i])+'-'+str(i))
	colorbar()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(speed_curves[i], label = str(shank[i])+'-'+str(i))
	legend()

show()

cc = compute_CrossCorrs(spikes, wake_ep, 5, 1000, norm = True)
cc = cc.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 5.0)


for k in range(1,4):
	figure()
	count = 0 
	for i in np.where(shank == 0)[0]:	
	    for j in np.where(shank == k)[0]:
	        subplot(5,5,count+1) 
	        plot(cc[(i,j)], label = str(i)+'/'+str(j)) 
	        legend() 
	        count+=1
	        # xticks([])
	        yticks([])
	        axvline(0)
	# tight_layout()
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200124B2'

episodes = ['sleep', 'wake']
events = ['1']


##################################################################################################
# LOADING DATA
##################################################################################################

spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory)


##################################################################################################
# ANGULAR TUNING CURVES
##################################################################################################
tuning_curves2 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep, 5, 100)
autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep, 5, 100)
velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=False)
speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)

tuning_curves2 = smoothAngularTuningCurves(tuning_curves2, 10, 2)
for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)

velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
		
##################################################################################################
# CROSS-CORR
##################################################################################################
cc_wak = compute_CrossCorrs(spikes, wake_ep, 5, 500)
cc_slp = compute_CrossCorrs(spikes, sleep_ep, 5, 500)

cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
cc_slp = cc_slp.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

############################################################################################### 
# PLOT
###############################################################################################
sys.exit()
count = 0
for i in np.where(shank == 0)[0]: 
	for k in range(1,4):
	    for j in np.where(shank == k)[0]: 
	        fig = figure(figsize = (20,12))
	        subplot(3,4,1)	        
	        plot(cc_wak[(i,j)], label = (i,j)) 
	        legend() 
	        subplot(3,4,2)
	        plot(cc_slp[(i,j)], label = (i,j))
	        legend()
	        subplot(3,4,3)
	        plot(autocorr_wake[i], label = 'POSUB_'+str(i)+' wake', color = 'black')
	        plot(autocorr_wake[j], label = 'LMN_'+str(j)+' wake', color = 'red')	        
	        title('wake')
	        legend()
	        subplot(3,4,4)	        
	        plot(autocorr_sleep[i], label = 'POSUB_'+str(i)+' sleep', color = 'black')
	        plot(autocorr_sleep[j], label = 'LMN_'+str(j)+' sleep', color = 'red')	        
	        title('sleep')
	        legend()
	        subplot(3,4,5,projection='polar')
	        plot(tuning_curves2[i], label = 'POSUB '+str(i)+' '+str(shank[i]), color = 'black')
	        legend()
	        subplot(3,4,9,projection='polar')
	        plot(tuning_curves2[j], label = 'LMN '+str(j)+' '+str(shank[j]), color = 'red')
	        legend()
	        subplot(3,4,6,projection='polar')
	        plot(tuning_curves[1][i], label = 'POSUB '+str(i)+' '+str(shank[i]), color = 'black')
	        legend()	        
	        subplot(3,4,10,projection='polar')
	        plot(tuning_curves[1][j], label = 'LMN '+str(j)+' '+str(shank[j]), color = 'red')
	        legend()	        
	        subplot(3,4,7)
	        plot(velo_curves[i], label = 'POSUB '+str(i)+' '+str(shank[i]), color = 'black')
	        legend()	        
	        subplot(3,4,11)
	        plot(velo_curves[j], label = 'LMN '+str(j)+' '+str(shank[j]), color = 'red')
	        legend()	        
	        subplot(3,4,8)
	        plot(speed_curves[i], label = 'POSUB '+str(i)+' '+str(shank[i]), color = 'black')
	        legend()	        
	        subplot(3,4,12)
	        plot(speed_curves[j], label = 'LMN '+str(j)+' '+str(shank[j]), color = 'red')
	        legend()	        
	        tight_layout()
	        savefig('../figures/A3000/A3004/A3004-200124B2/fig_'+str(count)+'.pdf')
	        close(fig)
	        count += 1

os.system("/snap/bin/pdftk ../figures/A3000/A3004/A3004-200124B2/fig_*.pdf cat output ../figures/A3000/A3004/A3004-200124B2/fig_all.pdf")


sys.exit()

figure()
for i in spikes:
	subplot(5,7,i+1, projection = 'polar')
	plot(tuning_curves2[i], label = str(shank[i]))
	legend()



figure()
for i in spikes:
	subplot(5,7,i+1, projection = 'polar')
	plot(tuning_curves[1][i], label = str(shank[i]))
	legend()


figure()
for i in spikes:
	subplot(5,7,i+1)
	plot(autocorr_wake[i], label = str(shank[i]))
	# plot(autocorr_sleep[i])
	legend()

figure()
for i in spikes:
	subplot(5,7,i+1)
	plot(velo_curves[i], label = str(shank[i]))
	legend()

figure()
for i in spikes:
	subplot(5,7,i+1)
	plot(speed_curves[i], label = str(shank[i]))
	legend()

count = 0 
for i in np.where(shank == 0)[0]: 
	for k in range(1,4):
	    for j in np.where(shank == k)[0]: 
	        subplot(8,10,count+1) 
	        plot(cc_wak[(i,j)], label = (i,j)) 
	        legend() 
	        count+=1




show()




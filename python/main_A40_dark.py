import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

# data_directory = '/mnt/DataGuillaume/LMN/A1410/A1410-200116A/A1410-200116A'
data_directory = '/mnt/LocalHDD/DTN/A4002/A4002-200121A'

episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
events = ['1', '3']


spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')
# acceleration						= loadAuxiliary(data_directory, 2)
# sleep_ep							= refineSleepFromAccel(acceleration, sleep_ep)


lght_ep 							= wake_ep.loc[[0]]
dark_ep 							= wake_ep.loc[[1]]

tuning_curves = {}
ahv_curves = {}
speed_curves = {}
cc_all = {}
auto_all = {}

for ep, name in zip([lght_ep, dark_ep, sleep_ep], ['light', 'dark', 'sleep']):
# for ep, name in zip([lght_ep, dark_ep], ['light', 'dark']):
	if name != 'sleep':
		tuning_curves[name] 			= computeLMNAngularTuningCurves(spikes, position['ry'], ep, 120)[0][1]
		tuning_curves[name]				= smoothAngularTuningCurves(tuning_curves[name], 20, 5)
		ahv_curves[name] 				= computeAngularVelocityTuningCurves(spikes, position['ry'], ep, nb_bins = 30, norm=False)
		ahv_curves[name] 				= ahv_curves[name].rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 2.0)
		speed_curves[name]				= computeSpeedTuningCurves(spikes, position[['x', 'z']], ep)
		speed_curves[name] 				= speed_curves[name].rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
	cc_all[name] 					= compute_CrossCorrs(spikes, ep, 5, 1000, norm = True)
	cc_all[name] 					= cc_all[name].rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 10.0)
	auto_all[name],_ 				= compute_AutoCorrs(spikes, ep)




ahv_diff = (ahv_curves['light']-ahv_curves['dark'])
speed_diff = (speed_curves['light']-speed_curves['dark'])

ahv_diff = (ahv_curves['light']-ahv_curves['dark'])/ahv_curves['dark']
speed_diff = (speed_curves['light']-speed_curves['dark'])/speed_curves['dark']
# cc_diff	= cc_all['light']-cc_all['dark']
# auto_diff = auto_all['light']-auto_all['dark']

all_diff = {'ahv':ahv_diff, 'speed_diff':speed_diff}


figure()
for i, n in enumerate(all_diff):
	subplot(1,2,i+1)	
	# plot(all_diff[n][idx], color = colors[j], alpha = 0.5)
	plot(all_diff[n].mean(1))
	x = all_diff[n].index.values
	fill_between(x, all_diff[n].mean(1) - all_diff[n].sem(1), all_diff[n].mean(1) + all_diff[n].sem(1), alpha = 0.4)
	legend()
	title('Light - Dark')
	
############################################################################################### 
# PLOT
###############################################################################################

figure()
ct = 0
for i in spikes:
	subplot(5,6,i+1, projection = 'polar')
	for name in tuning_curves:
		plot(tuning_curves[name][i], label = name)
	text(0.5, 0.9, str(shank[i][0])+"-"+str(ct), transform=gca().transAxes)	
	if i == 0: legend()
	ct += 1


figure()
ct = 0
for i in spikes:
	subplot(5,6,i+1)
	for name in tuning_curves:
		plot(ahv_curves[name][i], label = name)
	text(0.5, 0.9, str(shank[i][0])+"-"+str(ct), transform=gca().transAxes)
	ct += 1


figure()
ct = 0
for i in spikes:
	subplot(5,6,i+1)
	for name in tuning_curves:
		plot(speed_curves[name][i], label = name)
	text(0.5, 0.9, str(shank[i][0])+"-"+str(ct), transform=gca().transAxes)
	ct+=1

# grids = [[10,10],[6,7],[6,9]]

# for k in range(1,4):
# 	figure()
# 	count = 0 
# 	for i in np.where(shank == 0)[0]:	
# 	    for j in np.where(shank == k)[0]:
# 	        subplot(grids[k-1][0],grids[k-1][1],count+1) 
# 	        for name in cc_all:
# 	        	plot(cc_all[name][(i,j)], label = name)
# 	        if count == 0: legend()	        
# 	        title(str(i)+'/'+str(j))
# 	        count+=1

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
from matplotlib import gridspec
import sys
from scipy.ndimage.filters import gaussian_filter

# data_directory = '/mnt/DataGuillaume/LMN/A1407/A1407-190429'
# data_directory = '/mnt/DataGuillaume/LMN/A1407/A1407-190425'
# data_directory = '/mnt/DataGuillaume/PostSub/A3003/A3003-190516A'
#data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200117C/A3004-200117C'
# data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200122B'
# data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200122C'
# data_directory = '/mnt/DataGuillaume/LMN-POSTSUB/A3004/A3004-200124B2'

data_directory = '/mnt/Data2/LMN-PSB-2/A3013/A3013-210806A'

#episodes = ['sleep', 'wake']
episodes = ['sleep', 'wake', 'wake', 'sleep', 'wake', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']

events = ['1', '2', '4', '5']





spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					

tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[2]], 120)

tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 2)

tokeep, stat = findHDCells(tuning_curves, z = 5, p = 0.01, m = 0.3)

rip_ep, rip_tsd 					= loadRipples(data_directory)

mean_fr = computeMeanFiringRate(spikes, [wake_ep], ['wake'])

mean_wf, max_ch = loadMeanWaveforms(data_directory)

############################################################################################### 
# PLOT
###############################################################################################

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue']

shank = shank.flatten()

figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes))),count, projection = 'polar')
		plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		# plot(tuning_curves2[1][i], '--', color = colors[shank[i]-1])
		if i in tokeep:
			plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1], linewidth = 3)
		legend()
		count+=1
		gca().set_xticklabels([])


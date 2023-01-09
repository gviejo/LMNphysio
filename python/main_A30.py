import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
from matplotlib import gridspec
import sys
from scipy.ndimage.filters import gaussian_filter

data_directory = '/mnt/DataRAID2/LMN-PSB/A3026/A3026-221205A'

episodes = ['sleep', 'wake', 'sleep']
#episodes = ['sleep', 'wake', 'wake', 'sleep', 'wake', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']

events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					

tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 120)


spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 30)

velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=False)


tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 2)


velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

# CHECKING HALF EPOCHS
wake2_ep = splitWake(wake_ep.loc[[0]])
tokeep2 = []
stats2 = []
tcurves2 = []
for i in range(2):
	# tcurves_half = computeLMNAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])[0][1]
	tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
	tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)
	tokeep, stat = findHDCells(tcurves_half)
	tokeep2.append(tokeep)
	stats2.append(stat)
	tcurves2.append(tcurves_half)

tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

#tokeep = np.intersect1d(np.where(shank<=4)[0], tokeep)


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
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		# plot(tuning_curves2[1][i], '--', color = colors[shank[i]-1])
		# if i in tokeep:
		# 	plot(tuning_curves[1][i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1], linewidth = 3)
		legend()
		count+=1
		gca().set_xticklabels([])



figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count)
		tmp = spatial_curves[i].values
		tmp = gaussian_filter(tmp, 1)
		imshow(tmp, interpolation = 'bilinear', cmap = 'jet')	
		colorbar()
		count += 1


figure()
gs = gridspec.GridSpec(3, len(tokeep))
for i, n in enumerate(tokeep):	
	subplot(gs[0,i], projection = 'polar')
	plot(tuning_curves[n])
	xticks([])
	title(shank[n])
	subplot(gs[1,i])
	tmp = spatial_curves[n].values
	tmp = gaussian_filter(tmp, 1)
	imshow(tmp, interpolation = 'bilinear', cmap = 'jet')	
	colorbar()
	subplot(gs[2,i])
	plot(velo_curves[n])



sys.exit()

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
cc = cc.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


for k in range(1,8):
	figure()
	count = 0
	for i in np.where(shank <= 4)[0]:	
	    for j in np.where(shank > 4)[0]:
	        subplot(10,10,count+1) 
	        plot(cc[(i,j)], label = str(i)+'/'+str(j)) 
	        # legend() 
	        count+=1
	        # xticks([])
	        yticks([])
	        axvline(0)
	# tight_layout()




#tokeep = [5, 7, 9, 12, 16, 18, 20, 21, 23, 24, 29, 35, 39, 50]

tokeep = [3,4,5,7,10,12,16,18,20,21,23,24,26,27,29,31,32,35,37,38,39,41,50,52,59,62,63]

mean_frate = computeMeanFiringRate({n:spikes[n] for n in tokeep}, [wake_ep.loc[[i]] for i in range(4)], range(4))

tc = {}
for i in range(4):
	tuning_curves = computeAngularTuningCurves({n:spikes[n] for n in tokeep}, position['ry'], wake_ep.loc[[i]], 120)
	tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 2)
	tc[i] = tuning_curves

figure()
gs = GridSpec(4,4)
for i,j in combinations(range(4),2):
	subplot(gs[i,j])
	plot(tc[i].max(), tc[j].max(), 'o')	

show()

figure()
gs = GridSpec(4,4)
for i,j in combinations(range(4),2):
	subplot(gs[i,j])
	tmp = tc[i].max() - tc[j].max()
	plot(np.sort(tmp), 'o')	

show()

figure()
gs = GridSpec(4,4)
for i,j in combinations(range(4),2):
	subplot(gs[i,j])
	plot(mean_frate[i], mean_frate[j], 'o')

show()

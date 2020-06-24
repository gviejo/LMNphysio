import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

# data_directory = '/mnt/DataGuillaume/LMN/A1410/A1410-200116A/A1410-200116A'
# data_directory = '/mnt/LocalHDD/A1410-200121A/A1410-200121A'
# data_directory = '/mnt/DataGuillaume/LMN/A1410/A1410-200122A'
# data_directory = '/mnt/DataGuillaume/LMN-ADN/A5002/A5002-200224A'
# data_directory = '/mnt/DataGuillaume/LMN-ADN/A5004/A5004-200609A'
data_directory = '/mnt/DataGuillaume/LMN-ADN/A5007/A5007-200623A'

# data_directory = '../data/A1400/A1407/A1407-190422'
# data_directory = '/mnt/DataGuillaume/PostSub/A3003/A3003-190516A'

# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep', 'wake', 'wake', 'wake', 'sleep']
# episodes = ['wake', 'sleep']
events = ['1']
# events = ['1', '3', '4', '5']
# events = ['1', '3', '5']






spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
acceleration 						= acceleration[[0,1,2]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

# tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 61)
spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep.loc[[0]], 30)
autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
# autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)
velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=False)
# sys.exit()
# mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, sleep_ep], ['wake', 'sleep'])
speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)

accel_curves 						= computeAccelerationTuningCurves(spikes, position[['x', 'z']], wake_ep)

# downsampleDatFile(data_directory)

for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)

tokeep, stat = findHDCells(tuning_curves[1])



velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


		
cc1 = compute_CrossCorrs(spikes, wake_ep, 5, 1000, norm = False)
cc1 = cc1.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
cc2 = compute_CrossCorrs(spikes, wake_ep, 0.5, 200, norm = True)
cc2 = cc2.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

cc3 = compute_CrossCorrs(spikes, sleep_ep, 5, 300, norm = False)
cc3 = cc3.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
cc4 = compute_CrossCorrs(spikes, sleep_ep, 0.5, 200, norm = True)
cc4 = cc4.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


############################################################################################### 
# PLOT
###############################################################################################
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

shank = shank.flatten()

for j in np.unique(shank):
	figure()
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,k+1, projection = 'polar')
		plot(tuning_curves[1][i], label = str(shank[i]) + ' ' + str(i))
		if i in tokeep:
			plot(tuning_curves[1][i], label = str(shank[i]) + ' ' + str(i), linewidth = 3)
		legend()

for j in np.unique(shank):
	figure()
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,k+1)
		plot(velo_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]])
		# if i in tokeep:
		# 	plot(tuning_curves[1][i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]], linewidth = 3)
		# legend()

for j in np.unique(shank):
	figure()
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,k+1)
		imshow(spatial_curves[i], interpolation = 'bilinear')
		# if i in tokeep:
		# 	plot(tuning_curves[1][i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]], linewidth = 3)
		# legend()



sys.exit()





figure()
count = 0
for c, i in enumerate(tokeep):		
	subplot(int(np.sqrt(len(tokeep)))+1,int(np.sqrt(len(tokeep)))+1,c+1)
	plot(velo_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]])
	legend()

figure()
count = 0
for c, i in enumerate(tokeep):		
	# subplot(int(np.sqrt(len(tokeep)))+1,int(np.sqrt(len(tokeep)))+1,c+1)
	plot(velo_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]])
	# legend()


figure()
count = 0
for c, i in enumerate(tokeep):		
	subplot(int(np.sqrt(len(tokeep)))+1,int(np.sqrt(len(tokeep)))+1,c+1)
	plot(speed_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]])
	legend()



figure()
for i in spikes:
	subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,i+1)
	imshow(spatial_curves[i], interpolation = 'bilinear')
	title(str(shank[i]) + ' ' + str(i))
	colorbar()

wake_ep1 = nts.IntervalSet(start = wake_ep.loc[0, 'start'], end = wake_ep.loc[0, 'start']+wake_ep.tot_length()/2)
wake_ep2 = nts.IntervalSet(start = wake_ep.loc[0, 'start']+wake_ep.tot_length()/2, end = wake_ep.loc[0, 'end'])

spatial_curves1, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep1, 30)
spatial_curves2, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep2, 30)

from scipy import signal
from scipy.ndimage import gaussian_filter
figure()
i = 30
tmp = spatial_curves[i]
tmp2 = gaussian_filter(tmp, sigma=0.5)
subplot(121)
imshow(tmp2, interpolation = 'bilinear')
title(str(shank[i]) + ' ' + str(i))
colorbar()
subplot(122)
cr = signal.correlate2d(tmp2, tmp2)
imshow(cr)
show()

sys.exit()

figure()
for i in spikes:
	if i in tokeep:
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,i+1)
		plot(speed_curves[i], label = str(shank[i]), color = colors[shank[i]])
		legend()

show()





from matplotlib import gridspec
from itertools import product
# adn = list(np.where(shank == 1)[0]) + list(np.where(shank == 2)[0])
# lmn = list(np.where(shank > 2)[0])

adn = list(np.where(shank <= 5)[0])
lmn = list(np.where(shank >= 6)[0])

adn = np.intersect1d(adn, tokeep)
lmn = np.intersect1d(lmn, tokeep)


pairs = list(product(adn, lmn))

sys.exit()

for j, n in enumerate(adn):	
	figure()
	gs = gridspec.GridSpec(5, len(lmn))
	for i, m in enumerate(lmn):	
		subplot(gs[0,i], projection = 'polar')
		tmp = tuning_curves[1][m]
		tmp = tmp/tmp.max()
		plot(tmp)
		tmp = tuning_curves[1][n]
		tmp = tmp/tmp.max()
		plot(tmp)
		
		subplot(gs[1,i])
		plot(cc2[(n,m)], label = m)
		grid()

		subplot(gs[2,i])
		plot(cc4[(n,m)], color = 'red')
		grid()

		subplot(gs[3,i])
		plot(cc1[(n,m)], label = n)	
		grid()

		subplot(gs[4,i])
		plot(cc3[(n,m)], color = 'red')
		grid()

figure()
for i, (m, n) in enumerate(pairs):
	subplot(8,8,i+1)
	plot(cc1[(m,n)], label = str(m)+' '+str(n))


figure()
for i, (m, n) in enumerate(pairs):
	subplot(4,8,i+1)
	plot(cc2[(m,n)], label = str(m)+' '+str(n))

figure()
for i, (m, n) in enumerate(pairs):
	subplot(4,8,i+1)
	plot(cc3[(m,n)], color = 'red', label = str(m)+' '+str(n))
	legend()

figure()
for i, (m, n) in enumerate(pairs):
	subplot(4,8,i+1)
	plot(cc4[(m,n)], color = 'red', label = str(m)+' '+str(n))
	legend()

tcurves = tuning_curves[1][tokeep]
tcurves = tcurves/tcurves.max(0)

figure()
subplot(111,projection = 'polar')
plot(tcurves, alpha = 0.6)
plot(tcurves.mean(1), color = 'black', linewidth = 4)


from itertools import product
shank3 = np.where(shank == 3)[0]
shank4 = np.where(shank == 4)[0]
pairs = list(product(shank3, shank4))

p = (23, 17)

# p = (15, 17)
figure()
for i,p in enumerate(pairs):
	subplot(10,12,i+1)
	tmp1 = compute_PairsCrossCorr(spikes, wake_ep, p, 5, 2000, True)
	tmp1 = tmp1.rolling(window=20, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
	plot(tmp1)
# tmp2 = compute_PairsCrossCorr(spikes, sleep_ep, p, 0.5, 200, True)
# tmp2 = tmp2.rolling(window=20, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

p = (8,30)
tmp1 = compute_PairsCrossCorr(spikes, wake_ep, p, 5, 2000, True)
tmp1 = tmp1.rolling(window=20, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
tmp2 = compute_PairsCrossCorr(spikes, sleep_ep, p, 5, 2000, True)
tmp2 = tmp2.rolling(window=20, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
figure()
subplot(221, projection = 'polar')
tcurves = tuning_curves[1][list(p)]
tcurves = tcurves/tcurves.max()
plot(tcurves[p[0]])
plot(tcurves[p[1]])
subplot(222)
plot(tmp1, label = 'wake')
plot(tmp2, label = 'sleep')
legend()
subplot(223)
plot(velo_curves[p[0]])
plot(velo_curves[p[1]])
subplot(224)
plot(speed_curves[p[0]])
plot(speed_curves[p[1]])

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean


data_directory = '/mnt/DataGuillaume/LMN-ADN/A5002/A5002-200303B'


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

sys.exit()

# tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 61)
spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep.loc[[0]], 30)
autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
# autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)
velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], nb_bins = 30, norm=False)
# sys.exit()
# mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, sleep_ep], ['wake', 'sleep'])
speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep.loc[[0]])

accel_curves 						= computeAccelerationTuningCurves(spikes, position[['x', 'z']], wake_ep.loc[[0]])

for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)
velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

tokeep, stat = findHDCells(tuning_curves[1], z = 10, p = 0.001)


# downsampleDatFile(data_directory, n_channels)

sys.exit()




		
cc1 = compute_CrossCorrs(spikes, wake_ep, 5, 1000, norm = False)
cc1 = cc1.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
cc2 = compute_CrossCorrs(spikes, wake_ep, 0.20, 400, norm = True)
cc2 = cc2.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 2.0)

cc3 = compute_CrossCorrs(spikes, sleep_ep, 5, 300, norm = False)
cc3 = cc3.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
cc4 = compute_CrossCorrs(spikes, sleep_ep, 0.20, 400, norm = True)
cc4 = cc4.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 2.0)




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
		plot(tuning_curves[1][i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		# plot(tuning_curves2[1][i], '--', color = colors[shank[i]-1])
		if i in tokeep:
			plot(tuning_curves[1][i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1], linewidth = 3)
		legend()
		count+=1
		gca().set_xticklabels([])

				
figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count)
		plot(velo_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		# plot(velo_curves2[i], '--', color = colors[shank[i]-1])
		legend()
		count+=1
		gca().set_xticklabels([])


figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count)
		plot(speed_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		# plot(speed_curves2[i], '--', color = colors[shank[i]-1])
		legend()
		count+=1
		gca().set_xticklabels([])

figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count)
		imshow(spatial_curves[i], interpolation = 'bilinear')
		colorbar()
		count += 1




sys.exit()





from matplotlib import gridspec
from itertools import product
from itertools import combinations
# adn = list(np.where(shank == 1)[0]) + list(np.where(shank == 2)[0])
# lmn = list(np.where(shank > 2)[0])

adn = list(np.where(shank == 3)[0])
lmn = list(np.where(shank == 5)[0])

adn = np.intersect1d(adn, tokeep)
lmn = np.intersect1d(lmn, tokeep)

# lmn = list(np.where(shank == 2)[0])

pairs = list(product(adn, lmn))

# pairs = list(combinations(lmn))



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
		plot(cc1[(n,m)], label = n)	
		grid()

		subplot(gs[3,i])
		plot(cc4[(n,m)], color = 'red')
		grid()


		subplot(gs[4,i])
		plot(cc3[(n,m)], color = 'red')
		grid()


neurons = np.hstack((adn, lmn))
peaks = pd.Series(index=neurons,data = np.array([circmean(tuning_curves[1].index.values, tuning_curves[1][i].values) for i in neurons])).sort_values()

pairs = pd.Series(index = pairs, data = np.nan)
for i,j in pairs.index:	
	a = peaks[i] - peaks[j]
	pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))


pairs = pairs.dropna().sort_values()



# CROSS CORR

figure()
# angular differences
subplot(231)
plot(pairs.values, np.arange(len(pairs))[::-1])
xticks([0, np.pi], ['0', r'180'])
yticks([0, len(pairs)-1], [len(pairs), 1])
xlabel("Ang. diff.", labelpad = -0.5)
ylabel("Pairs", labelpad = -20)
ylim(0, len(pairs)-1)

subplot(2,3,2)
tmp = cc1[pairs.index].T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 2), aspect = 'auto', cmap = 'jet')
title('wake')
xticks([0, np.where(cc1.index.values == 0)[0][0], len(cc1)], [cc1.index[0], 0, cc1.index[-1]])

subplot(2,3,3)
tmp = cc3[pairs.index].T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 1), aspect = 'auto', cmap = 'jet')
title('sleep')
xticks([0, np.where(cc3.index.values == 0)[0][0], len(cc3)], [cc3.index[0], 0, cc3.index[-1]])

subplot(2,3,5)
plot(cc1, alpha = 0.5)

subplot(2,3,6)
plot(cc3, alpha = 0.5)


figure()
# angular differences
subplot(231)
plot(pairs.values, np.arange(len(pairs))[::-1])
xticks([0, np.pi], ['0', r'180'])
yticks([0, len(pairs)-1], [len(pairs), 1])
xlabel("Ang. diff.", labelpad = -0.5)
ylabel("Pairs", labelpad = -20)
ylim(0, len(pairs)-1)

subplot(2,3,2)
tmp = cc2[pairs.index].T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 2), aspect = 'auto', cmap = 'jet')
title('wake')
xticks([0, np.where(cc2.index.values == 0)[0][0], len(cc2)], [cc2.index[0], 0, cc2.index[-1]])

subplot(2,3,3)
tmp = cc4[pairs.index].T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 1), aspect = 'auto', cmap = 'jet')
title('sleep')
xticks([0, np.where(cc4.index.values == 0)[0][0], len(cc4)], [cc4.index[0], 0, cc4.index[-1]])

subplot(2,3,5)
plot(cc2, alpha = 0.5)

subplot(2,3,6)
plot(cc4, alpha = 0.5)



tmp = (cc2[pairs.index] - cc2[pairs.index].mean())/cc2[pairs.index].std()

tmp2 = (cc4[pairs.index] - cc4[pairs.index].mean())/cc4[pairs.index].std()

plot(tmp2.iloc[:,0:5].mean(1))
plot(tmp2.iloc[:,-5:].mean(1)) 
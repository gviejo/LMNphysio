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
data_directory = '/mnt/DataGuillaume/LMN-ADN/A5001/A5001-200226A/A5001-200226A'

# data_directory = '../data/A1400/A1407/A1407-190422'
# data_directory = '/mnt/DataGuillaume/PostSub/A3003/A3003-190516A'

# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake']
episodes = ['sleep', 'wake']
# episodes = ['wake', 'sleep']
# events = ['1', '3']
events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
if 'A5002' in data_directory:
	acceleration 						= acceleration[[0,1,2]]
else:
	acceleration 						= acceleration[[3,4,5]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

# tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 20)
autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
# autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)
velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=False)
# sys.exit()
# mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, sleep_ep], ['wake', 'sleep'])
speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)

# downsampleDatFile(data_directory)

for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)

velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
		
cc1 = compute_CrossCorrs(spikes, wake_ep, 5, 1000, norm = False)
cc1 = cc1.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
cc2 = compute_CrossCorrs(spikes, wake_ep, 0.5, 200, norm = True)
cc2 = cc2.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

cc3 = compute_CrossCorrs(spikes, sleep_ep, 5, 1000, norm = False)
cc3 = cc3.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
cc4 = compute_CrossCorrs(spikes, sleep_ep, 0.5, 200, norm = True)
cc4 = cc4.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


############################################################################################### 
# PLOT
###############################################################################################


for i, s in enumerate(np.unique(shank)):
	figure()
	neurons = np.where(shank == s)[0]
	for j, n in enumerate(neurons):
		subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,j+1, projection = 'polar')
		plot(tuning_curves[1][n], label = str(shank[n]))
		# legend()

for i, s in enumerate(np.unique(shank)):
	figure()
	neurons = np.where(shank == s)[0]
	for j, n in enumerate(neurons):
		subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,j+1)
		plot(velo_curves[n], label = str(shank[n]))
		# legend()

for i, s in enumerate(np.unique(shank)):
	figure()
	neurons = np.where(shank == s)[0]
	for j, n in enumerate(neurons):
		subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,j+1)
		plot(speed_curves[n], label = str(shank[n]))
		# legend()





# for k, s in enumerate(np.unique(shank)[1:]):
# 	figure()
# 	count = 0 
# 	for i in np.where(shank == 6)[0]:	
# 	    for j in np.where(shank == s)[0]:
# 	        subplot(3,3,count+1) 
# 	        plot(cc2[(i,j)], label = str(i)+'/'+str(j)) 
# 	        legend() 
# 	        count+=1
# 	        # xticks([])
# 	        yticks([])
# 	        # axvline(0)
# 	# tight_layout()



from matplotlib import gridspec
from itertools import product
adn = list(np.where(shank == 1)[0]) + list(np.where(shank == 2)[0])
lmn = list(np.where(shank > 2)[0])

adn = list(np.where(shank == 6)[0])
lmn = list(np.where(shank > 6)[0])



pairs = list(product(adn, lmn))


for j, n in enumerate(lmn):	
	figure()
	gs = gridspec.GridSpec(5, len(adn))
	for i, m in enumerate(adn):	
		subplot(gs[0,i], projection = 'polar')
		tmp = tuning_curves[1][m]
		tmp = tmp/tmp.max()
		plot(tmp)
		tmp = tuning_curves[1][n]
		tmp = tmp/tmp.max()
		plot(tmp)
		
		subplot(gs[1,i])
		plot(cc2[(m,n)], label = m)
		
		subplot(gs[2,i])
		plot(cc4[(m,n)], color = 'red')

		subplot(gs[3,i])
		plot(cc1[(m,n)], label = n)	

		subplot(gs[4,i])
		plot(cc3[(m,n)], color = 'red')


figure()
for i, (m, n) in enumerate(pairs):
	subplot(5,10,i+1)
	plot(cc1[(m,n)])


figure()
for i, (m, n) in enumerate(pairs):
	subplot(5,10,i+1)
	plot(cc2[(m,n)])

figure()
for i, (m, n) in enumerate(pairs):
	subplot(5,10,i+1)
	plot(cc3[(m,n)], color = 'red')

figure()
for i, (m, n) in enumerate(pairs):
	subplot(5,10,i+1)
	plot(cc4[(m,n)], color = 'red')

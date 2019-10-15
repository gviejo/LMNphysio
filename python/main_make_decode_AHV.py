import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle

from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


# data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

sessions = ['A1407-190411', 'A1407-190416', 'A1407-190417', 'A1407-190422']

for s in sessions:
	path = os.path.join(data_directory, s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	episodes 							= info.filter(like='Trial').loc[s].dropna().values
	events								= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sleep_ep 							= loadEpoch(path, 'sleep')					
	# sws_ep								= loadEpoch(path, 'sws')
	# rem_ep								= loadEpoch(path, 'rem')

	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################

	neurons = [0, 9, 10, 12, 14]


	####################################################################################################################
	# MAKE VELOCITY
	####################################################################################################################
	bin_size 		= 10 # ms
	time_bins		= np.arange(wake_ep.loc[0,'start'], wake_ep.loc[0, 'end']+bin_size, bin_size*1000) # assuming microseconds

	angle 			= position['ry']	
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))	
	tmp2 			= tmp.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=30.0)
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp3 			= tmp2.groupby(index).mean()
	tmp3.index 		= time_bins[np.unique(index)-1]+bin_size/2 * 1000
	tmp3 			= nts.Tsd(tmp3)	
	tmp4			= np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
	velocity 		= nts.Tsd(t=tmp3.index.values[1:], d = tmp4)
	
	####################################################################################################################
	# COMPUTE TUNING CRUVES
	####################################################################################################################	
	nb_bins			= 31
	bins 			= np.linspace(-np.pi, np.pi, nb_bins)
	idx 			= bins[0:-1]+np.diff(bins)/2
	velo_curves		= pd.DataFrame(index = idx, columns = neurons)

	for k in neurons:
		spks 		= spikes[k].restrict(wake_ep)		
		speed_spike = velocity.realign(spks)
		spike_count, bin_edges = np.histogram(speed_spike, bins)
		occupancy, _ = np.histogram(velocity.restrict(wake_ep), bins)
		spike_count = spike_count/(occupancy+1)
		velo_curves[k] = spike_count*(1/(bin_size*1e-3))
		# normalizing by firing rate 
		velo_curves[k] = velo_curves[k]/(len(spikes[k].restrict(wake_ep))/wake_ep.tot_length('s'))

	velo_curves2 = velo_curves.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 2.0)

	############################################################################################### 
	# DECODING AHV
	###############################################################################################
	for i, n in enumerate(neurons):
		subplot(2,3,i+1)
		plot(velo_curves2[n])

	show()

	sys.exit()

############################################################################################### 
# PLOT
###############################################################################################


figure()
for i in spikes:
	subplot(3,5,i+1, projection = 'polar')
	plot(tuning_curves[1][i], label = str(shank[i]))
	legend()
show()



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
	plot(autocorr_sleep[i])
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(velo_curves[i], label = str(shank[i]))
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	imshow(spatial_curves[i])
	colorbar()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(speed_curves[i], label = str(shank[i]))
	legend()

show()



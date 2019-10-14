import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys, os
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec

data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

# sessions = ['A1407-190416', 'A1407-190417']
sessions = info.loc['A1407-190411':].index.values
# sessions = info.index.values[1:]

allvcurves = {}
allvcurves2 = {}


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
	# velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 50, bin_size = 10000)	
	angle 			= position['ry']
	ep 				= wake_ep
	nb_bins 		= 30
	bin_size 		= 10000

	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=30.0)
	time_bins		= np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
	# index 			= np.digitize(tmp2.index.values, time_bins)
	# tmp3 			= tmp2.groupby(index).mean()
	# tmp3.index 		= time_bins[np.unique(index)-1]+50000
	# tmp3 			= nts.Tsd(tmp3)
	# tmp4			= np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
	tmp2 			= nts.Tsd(tmp2)
	tmp4			= np.diff(tmp2.values)/np.diff(tmp2.as_units('s').index.values)	
	velocity 		= nts.Tsd(t=tmp2.index.values[1:], d = tmp4)
	velocity 		= velocity.restrict(ep)	
	bins 			= np.linspace(-np.pi, np.pi, nb_bins)
	idx 			= bins[0:-1]+np.diff(bins)/2
	velo_curves		= pd.DataFrame(index = idx, columns = np.arange(len(spikes)))
	# refining with low linear speed
	# xy 			= position[['x', 'z']]
	# index 		= np.digitize(xy.index.values, time_bins)
	# tmp 		= xy.groupby(index).mean()
	# tmp.index 	= time_bins[np.unique(index)-1]+bin_size/2	
	# distance	= np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))	
	# speed 		= nts.Tsd(t = tmp.index.values[0:-1]+ bin_size/2, d = distance/(bin_size*1e-6))
	# speed 		= speed.restrict(ep)
	# speed2 		= speed.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 5.0)
	# idx 		= (speed2 < 0.05)*1
	# start 		= np.where(np.diff(idx) == 1)[0]
	# end 		= np.where(np.diff(idx) == -1)[0]
	# if start[0] > end[0]:
	# 	start = np.hstack((np.zeros(1), start))
	# if start[-1] > end[-1]:
	# 	end = np.hstack((end, len(idx)-1))	
	# new_ep = nts.IntervalSet(start = idx.index.values[start.astype('int')], end = idx.index.values[end.astype('int')])

	# plot(speed2)
	# [plot([new_ep.iloc[i,0], new_ep.iloc[i,1]], [-0.1, -0.1]) for i in new_ep.index.values]
	
	for k in spikes:
		spks 		= spikes[k]
		spks 		= spks.restrict(ep)
		speed_spike = velocity.realign(spks)
		spike_count, bin_edges = np.histogram(speed_spike, bins)
		occupancy, _ = np.histogram(velocity.restrict(ep), bins)
		spike_count = spike_count/(occupancy+1)
		velo_curves[k] = spike_count*(1/(bin_size*1e-6))


	velo_curves2 						= velo_curves.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


	allvcurves[s] = velo_curves
	allvcurves2[s] = velo_curves2



	print(s)
# for s in sessions:
	figure(figsize = (20,15))	
	for j, n in enumerate(allvcurves[s].columns):
		subplot(5,7,j+1)
		plot(allvcurves[s][n])
		plot(allvcurves2[s][n])
		title(n)
	show()
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys, os
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from umap import UMAP
from sklearn.decomposition import PCA
import _pickle as cPickle

data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

# sessions = ['A1407-190416', 'A1407-190417']
sessions = info.loc['A1407-190403':].index.values
# sessions = info.index.values[1:]

allvcurves = {}
allvcurves2 = {}

allahv = []

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
	nb_bins 		= 31
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
		# normalizing by firing rate 
		# velo_curves[k] = velo_curves[k]/(len(spikes[k].restrict(wake_ep))/wake_ep.tot_length('s'))


	velo_curves2 						= velo_curves.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 2.0)


	allvcurves[s] = velo_curves
	allvcurves2[s] = velo_curves2

	velo_curves2.columns = pd.Index([s+'_'+str(n) for n in velo_curves2.columns.values])

	allahv.append(velo_curves2)


	# print(s)
	# figure(figsize = (20,15))	
	# for j, n in enumerate(allvcurves[s].columns):
	# 	subplot(5,7,j+1)
	# 	plot(allvcurves[s][n])
	# 	plot(allvcurves2[s][n])
	# 	title(n)
	# show()

allahv = pd.concat(allahv, 1)


tmp1 = allahv.loc[-2:2].values
tmp1 = tmp1 - tmp1.mean(0)
# tmp1 = tmp1 / tmp1.std(0)

# ADDING AD AHV TUNING CURVES
AD_data = cPickle.load(open('../figures/figures_poster_2019/Data_AD_Mouse32-140822.pickle', 'rb'))

ad_ahv = AD_data['ahvcurves']
ad_ahv = ad_ahv.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
tmp2 = ad_ahv.loc[-2:2].values
tmp2 = tmp2 - tmp2.mean(0)
# tmp2 = tmp2 / tmp2.std(0)

tmp = np.hstack((tmp1, tmp2))

nucleus = np.hstack((np.zeros(tmp1.shape[1]), np.ones(tmp2.shape[1])))

ump = UMAP(n_neighbors = 10, n_components = 2, min_dist = 0.5).fit_transform(tmp.T)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 5, random_state = 0).fit(ump)

labels = kmeans.labels_

colors = np.array(['red', 'blue', 'orange', 'green', 'purple'])

figure()
subplot(231)
scatter(ump[:,0], ump[:,1], c = colors[kmeans.labels_])
for i,j in zip(range(5), range(5)):
	subplot(2,3,i+2)
	plot(tmp[:,labels == j], color = colors[j])

show()

#######################################
# SAVING
allahv.to_hdf('../figures/figures_poster_2019/allahv.h5', 'w')

ump = pd.DataFrame(index = allahv.columns, data = ump)

ump['label'] = labels

ump.to_hdf('../figures/figures_poster_2019/ump.h5', 'w')

tmp = pd.DataFrame(index = allahv.loc[-2:2].index.values, columns = allahv.columns, data = tmp)

tmp.to_hdf('../figures/figures_poster_2019/allahv_normalized.h5', 'w')
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

data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

# sessions = ['A1407-190416', 'A1407-190417']
sessions = info.loc['A1407-190403':].index.values
# sessions = info.index.values[1:]

allvcurves = {}
allvcurves2 = {}

allscurves = {}
allscurves2 = {}


allahv = []
allspd = []

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
	# COMPUTING AHV TUNING CURVES
	###############################################################################################	
	angle 			= position['ry']
	ep 				= wake_ep
	nb_bins 		= 31
	bin_size 		= 10000

	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=30.0)
	time_bins		= np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp2 			= tmp2.groupby(index).mean() # down to 100Hz
	tmp2.index 		= (time_bins[np.unique(index)-1] + (bin_size)/2).astype('int')
	tmp2 			= nts.Tsd(tmp2)
	tmp4			= np.diff(tmp2.values)/np.diff(tmp2.as_units('s').index.values)	
	velocity 		= nts.Tsd(t=tmp2.index.values[0:-1]+np.diff(tmp2.index.values)/2, d = tmp4)
	velocity 		= velocity.restrict(ep)	
	bins 			= np.linspace(-np.pi, np.pi, nb_bins)
	idx 			= bins[0:-1]+np.diff(bins)/2
	velo_curves		= pd.DataFrame(index = idx, columns = np.arange(len(spikes)))
	
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
	velo_curves.columns = pd.Index([s+'_'+str(n) for n in velo_curves.columns.values])
	velo_curves2.columns = pd.Index([s+'_'+str(n) for n in velo_curves2.columns.values])

	allahv.append(velo_curves2)

	############################################################################################### 
	# COMPUTING SPEED TUNING CURVES
	###############################################################################################	
	speed_max 	= 0.25

	xy 			= position[['x', 'z']]	
	index 		= np.digitize(xy.index.values, time_bins)
	xy 	 		= xy.groupby(index).mean() # down to 100Hz
	xy.index 	= (time_bins[np.unique(index)-1] + (bin_size)/2).astype('int')
	xy 			= xy.rolling(window=100, win_type='gaussian', center = True, min_periods = 1).mean(std=10)
	
	distance	= np.sqrt(np.power(np.diff(xy['x']), 2) + np.power(np.diff(xy['z']), 2))	
	speed 		= nts.Tsd(t = xy.index.values[0:-1]+np.diff(xy.index.values)/2, d = distance/(bin_size*1e-6))
	speed 		= speed.restrict(wake_ep)	
	bins 		= np.linspace(0, speed_max, nb_bins)
	idx 		= bins[0:-1]+np.diff(bins)/2
	speed_curves = pd.DataFrame(index = idx,columns = np.arange(len(spikes)))
	for k in spikes:
		spks 	= spikes[k].restrict(ep)		
		speed_spike = speed.realign(spks)
		spike_count, bin_edges = np.histogram(speed_spike, bins)
		occupancy, _ = np.histogram(speed, bins)
		spike_count = spike_count/(occupancy+1)
		speed_curves[k] = spike_count*(1/(bin_size*1e-6))

	speed_curves2 						= speed_curves.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 2.0)
	allscurves[s] = speed_curves
	allscurves2[s] = speed_curves2
	speed_curves.columns = pd.Index([s+'_'+str(n) for n in speed_curves.columns.values])
	speed_curves2.columns = pd.Index([s+'_'+str(n) for n in speed_curves2.columns.values])

	allspd.append(speed_curves2)

	
	# CORRELATION AHV SPEED

	speed_data = pd.concat([velocity.as_series().abs(), speed.as_series()*100], 1)
	speed_data.columns = pd.Index(['ahv', 'lsp'])

	# dowsample to 10 Hz
	time_bins		= np.arange(speed_data.index[0], speed_data.index[-1]+10000, 10000) # assuming microseconds	
	speed_data 		= speed_data.groupby(np.digitize(speed_data.index.values, time_bins)).mean() # down to 100Hz

	speed_data 	= speed_data[speed_data['lsp'] < 25]
	speed_data = speed_data[speed_data['ahv'] < np.pi]

	# scatter(speed_data['ahv'], speed_data['lsp'], alpha = 0.5, linewidth = 0)
	# xlabel("|AHV|")
	# ylabel("L speed")

	# show()

	sys.exit()	

	# print(s)
	# figure(figsize = (20,15))	
	# for j, n in enumerate(allvcurves[s].columns):
	# 	subplot(5,7,j+1)
	# 	plot(allvcurves[s][n])
	# 	plot(allvcurves2[s][n])
	# 	title(n)


	# figure(figsize = (20,15))	
	# for j, n in enumerate(allscurves[s].columns):
	# 	subplot(5,7,j+1)
	# 	plot(allscurves[s][n])
	# 	plot(allscurves2[s][n])
	# 	title(n)
	# show()

	# sys.exit()

allahv = pd.concat(allahv, 1)
allspd = pd.concat(allspd, 1)


tmp1 = allahv.loc[-2:2].values
idx1 = allahv.loc[-2:2].index.values
tmp1 = tmp1 - tmp1.mean(0)
tmp1 = tmp1 / tmp1.std(0)

tmp2 = allspd.values
idx2 = allspd.index.values
tmp2 = tmp2 - tmp2.mean(0)
tmp2 = tmp2 / tmp2.std(0)

tmp = np.hstack((tmp1.T, tmp2.T))

ump1 = UMAP(n_neighbors = 10, n_components = 2, min_dist = 0.5).fit_transform(tmp1.T)
ump2 = UMAP(n_neighbors = 10, n_components = 2, min_dist = 0.5).fit_transform(tmp2.T)
ump3 = UMAP(n_neighbors = 10, n_components = 2, min_dist = 0.5).fit_transform(tmp)

from sklearn.cluster import KMeans

kmeans1 = KMeans(n_clusters = 5, random_state = 0).fit(ump1)
kmeans2 = KMeans(n_clusters = 5, random_state = 0).fit(ump2)
kmeans3 = KMeans(n_clusters = 2, random_state = 0).fit(ump3)

labels1 = kmeans1.labels_
labels2 = kmeans2.labels_
labels3 = kmeans3.labels_

colors = np.array(['red', 'blue', 'orange', 'green', 'purple'])

figure()
subplot(231)
scatter(ump1[:,0], ump1[:,1], c = colors[labels1])
for i,j in zip(range(5), range(5)):
	subplot(2,3,i+2)
	plot(idx1, tmp1[:,labels1 == j], color = colors[j])

figure()
subplot(231)
scatter(ump2[:,0], ump2[:,1], c = colors[labels2])
for i,j in zip(range(5), range(5)):
	subplot(2,3,i+2)
	plot(idx2, tmp2[:,labels2 == j], color = colors[j])

figure()
scatter(ump3[:,0], ump3[:,1], c = colors[labels3])
figure()
ct = 1
for i,j in zip(range(2), range(2)):
	subplot(2,2,ct)
	plot(idx1, tmp1[:,labels3 == j], color = colors[j])
	ct+=1
for i,j in zip(range(2), range(2)):
	subplot(2,2,ct)
	plot(idx2, tmp2[:,labels3 == j], color = colors[j])
	ct+=1


show()

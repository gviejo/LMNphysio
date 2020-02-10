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
from pycircstat.descriptive import mean as circmean

data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

# sessions = ['A1407-190416', 'A1407-190417']
sessions = info.loc['A1407-190403':].index.values
# sessions = info.index.values[1:]

lmn_ahv = []
lmn_hdc = []

lmn_hd_info = []

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
	ahv_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, 61)
	hd_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)

	tokeep, stat 					= findHDCells(hd_curves)	

	frate 							= pd.Series(index = spikes.keys(), data = [len(spikes[k].restrict(wake_ep))/wake_ep.tot_length('s') for k in spikes.keys()])
	hd_curves	 					= hd_curves/frate

	names 							= pd.Index([s+'_'+str(n) for n in spikes.keys()])
	ahv_curves.columns 				= names
	hd_curves.columns				= names
	hdinfo 							= pd.Series(index=names, data = 0)
	hdinfo.loc[names[tokeep]] 		= 1
	lmn_ahv.append(ahv_curves)
	lmn_hdc.append(hd_curves)
	lmn_hd_info.append(hdinfo)

	
lmn_ahv = pd.concat(lmn_ahv, 1)
lmn_hdc = pd.concat(lmn_hdc, 1)
lmn_hd_info = pd.concat(lmn_hd_info, 0)

# ADDING AD AHV TUNING CURVES
AD_data = cPickle.load(open('../figures/figures_poster_2019/Data_AD_normalized.pickle', 'rb'))
adn_ahv = AD_data['ahvcurves']
adn_hdc = AD_data['tcurves']

# SMOOTHING
lmn_hdc = smoothAngularTuningCurves(lmn_hdc, window = 20, deviation = 3.0)
adn_hdc = smoothAngularTuningCurves(adn_hdc, window = 20, deviation = 3.0)

lmn_ahv = lmn_ahv.loc[-2.5:2.5].rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
adn_ahv = adn_ahv.loc[-2.5:2.5].rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)


# NEED TO CENTER TUNING CURVES
peaks = pd.Series(index=lmn_hdc.columns,data = np.array([circmean(lmn_hdc.index.values, lmn_hdc[i].values) for i in lmn_hdc.columns]))
for n in peaks.index:
	tmp = lmn_hdc[n].copy()
	new_index = tmp.index.values - peaks[n]
	new_index[new_index<-np.pi] += 2*np.pi
	new_index[new_index>np.pi] -= 2*np.pi
	tmp.index = pd.Index(new_index)
	tmp = tmp.sort_index()
	lmn_hdc[n] = tmp.values

peaks = pd.Series(index=adn_hdc.columns,data = np.array([circmean(adn_hdc.index.values, adn_hdc[i].values) for i in adn_hdc.columns]))
for n in peaks.index:
	tmp = adn_hdc[n].copy()
	new_index = tmp.index.values - peaks[n]
	new_index[new_index<-np.pi] += 2*np.pi
	new_index[new_index>np.pi] -= 2*np.pi
	tmp.index = pd.Index(new_index)
	tmp = tmp.sort_index()
	adn_hdc[n] = tmp.values


ad_neurons = adn_hdc.columns
lm_neurons = lmn_hdc.columns

# lmn_data = np.vstack((lmn_ahv.loc[-2.1:2.1,lm_neurons].values, lmn_hdc[lm_neurons].values))
# ad_data  = np.vstack((adn_ahv.loc[-2.1:2.1,ad_neurons].values, adn_hdc[ad_neurons].values))
# tmp = np.hstack((lmn_data, ad_data))
# tmp = np.hstack((lmn_data, ad_data))
tmp = np.hstack((lmn_ahv.loc[-2.1:2.1,lm_neurons].values,adn_ahv.loc[-2.1:2.1,ad_neurons].values))

nuc = np.hstack((np.zeros(len(lm_neurons)), np.ones(len(ad_neurons))))

ump = UMAP(n_neighbors = 40, n_components = 2, min_dist = 0.5).fit_transform(tmp.T)
scatter(ump[:,0], ump[:,1], c = nuc)
show()

sys.exit()

from sklearn.cluster import KMeans

tmp2 = lmn_ahv.values

ump2 = UMAP(n_neighbors = 10, n_components = 2, min_dist = 1).fit_transform(tmp.T)

labels = KMeans(n_clusters = 5, random_state = 0).fit(ump2).labels_

colors = np.array(['red', 'blue', 'orange', 'green', 'purple'])

figure()
subplot(231)
scatter(ump2[:,0], ump2[:,1], c = colors[kmeans.labels_])
for i,j in zip(range(5), range(5)):
	subplot(2,3,i+2)
	plot(tmp[:,labels == j], color = colors[j])

show()



#######################################
# SAVING
# allahv.to_hdf('../figures/figures_poster_2019/allahv.h5', 'w')

ump = pd.DataFrame(index = np.hstack((lm_neurons, ad_neurons)), data = ump)

ump['nucleus'] = ['lmn']*len(lm_neurons) + ['ad']*len(ad_neurons)

ump['hd'] = list(lmn_hd_info[lm_neurons].values) + [1]*len(ad_neurons)

ump['labels'] = labels

ump.to_hdf('../figures/figures_poster_2019/ump.h5', 'w')

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from sklearn.manifold import TSNE

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets_DTN = np.loadtxt(os.path.join(data_directory,'datasets_DTN.list'), delimiter = '\n', dtype = str, comments = '#')
infos_DTN = getAllInfos(data_directory, datasets_DTN)
datasets_LMN = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
infos_LMN = getAllInfos(data_directory, datasets_LMN)
datasets = np.hstack((datasets_LMN, datasets_DTN))
infos = {**infos_DTN, **infos_LMN}


# allauto = {e:[] for e in ['wak', 'rem', 'sws']}
# allfrates = {e:[] for e in ['wak', 'rem', 'sws']}
allauto = {e:[] for e in ['wak']}
allfrates = {e:[] for e in ['wak']}

hd_index = []
shanks_index = []

for s in datasets:
	print(s)
	name = s.split('/')[-1]
	path = os.path.join(data_directory, s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	episodes 							= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	episodes[episodes != 'sleep'] 		= 'wake'
	events								= list(np.where(episodes != 'sleep')[0].astype('str'))	
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sleep_ep 							= loadEpoch(path, 'sleep')					
	# sws_ep								= loadEpoch(path, 'sws')
	# rem_ep								= loadEpoch(path, 'rem')

	# Only taking the first wake ep
	wake_ep = wake_ep.loc[[0]]

	# # Taking only neurons from LMN
	if 'A50' in s:
		# spikes = {n:spikes[n] for n in np.intersect1d(np.where(shank.flatten()>2)[0], np.where(shank.flatten()<4)[0])}
		spikes = {n:spikes[n] for n in np.where(shank.flatten()>2)[0]}
	# 	neurons = [name+'_'+str(n) for n in spikes.keys()]
	# 	shanks_index.append(pd.Series(index = neurons, data = shank[shank>2].flatten()))
	# else:
	# 	neurons = [name+'_'+str(n) for n in spikes.keys()]
	# 	shanks_index.append(pd.Series(index = neurons, data = shank.flatten()))
	
	

	
	# ############################################################################################### 
	# # COMPUTING TUNING CURVES
	# ###############################################################################################
	tuning_curves = {1:computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)}
	for i in tuning_curves:
		tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)

	# CHECKING HALF EPOCHS
	wake2_ep = splitWake(wake_ep)
	tokeep2 = []
	stats2 = []
	tcurves2 = []
	for i in range(2):
		tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])
		tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
		tokeep, stat = findHDCells(tcurves_half)
		tokeep2.append(tokeep)
		stats2.append(stat)
		tcurves2.append(tcurves_half)

	tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
	tokeep2 = np.union1d(tokeep2[0], tokeep2[1])

	# figure()
	# for i, n in enumerate(tokeep2):
	# 	subplot(5,6,i+1, projection = 'polar')
	# 	plot(tcurves2[0][n])
	# 	plot(tcurves2[1][n])
	# 	if n in tokeep:
	# 		plot(tuning_curves[1][n], color = 'red')


	spikes = {n:spikes[n] for n in tokeep}
	neurons = [name+'_'+str(n) for n in spikes.keys()]
	hd_index.append([n for n in neurons if int(n.split('_')[1]) in tokeep])

	############################################################################################### 
	# COMPUTE AUTOCORRS
	###############################################################################################
	# for e, ep, tb, tw in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep], [10, 10, 1], [1000, 1000, 1000]):
	for e, ep, tb, tw in zip(['wak'], [wake_ep], [1], [1000]):
		autocorr, frates = compute_AutoCorrs(spikes, ep, tb, tw)
		autocorr.columns = pd.Index(neurons)
		frates.index = pd.Index(neurons)
		allauto[e].append(autocorr)
		allfrates[e].append(frates)

for e in allauto.keys():
	allauto[e] = pd.concat(allauto[e], 1)
	allfrates[e] = pd.concat(allfrates[e])

frates = pd.DataFrame.from_dict(allfrates)

allindex = []
data = []
halfauto = {}
for e in allauto.keys():
	# 1. starting at 2
	auto = allauto[e].loc[0.5:]
	# 3. lower than 100 
	auto = auto.drop(auto.columns[auto.apply(lambda col: col.max() > 100.0)], axis = 1)
	# # 4. gauss filt
	auto = auto.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
	auto = auto.loc[2:400]
	# Drop nan
	auto = auto.dropna(1)
	halfauto[e] = auto
	allindex.append(auto.columns)

hd_index = np.hstack(hd_index)
neurons = hd_index
# neurons = np.intersect1d(np.intersect1d(allindex[0], allindex[1]), allindex[2])
# neurons = np.intersect1d(neurons, fr_index)

# shanks_index = pd.concat(shanks_index)

data = np.hstack([halfauto[e][neurons].values.T for e in halfauto.keys()])

from umap import UMAP
from sklearn.cluster import KMeans

hd_index = np.intersect1d(neurons, hd_index)
c = pd.Series(index = neurons, data = 0)
c.loc[hd_index] = 1


# X = TSNE(2, 100).fit_transform(data)

U = UMAP(n_neighbors = 5, n_components = 2).fit_transform(data)
K = KMeans(n_clusters = 2, random_state = 0).fit(U).labels_

LMN = np.array([i for i,n in enumerate(neurons) if 'A14' in n])
DTN = np.array([i for i,n in enumerate(neurons) if 'A40' in n])


figure()
plot(U[:,0], U[:,1], 'o', color = 'grey', markersize = 10)
plot(U[LMN][:,0], U[LMN][:,1], 'o', color = 'red', markersize = 5, label = 'LMN')
plot(U[DTN][:,0], U[DTN][:,1], 'o', color = 'blue', markersize = 5, label = 'DTN')
legend()



figure()
for i in range(2):
	subplot(2,2,i+1)
	plot(allauto['wak'][neurons[K==i]], color = 'grey', alpha = 0.5)
	subplot(2,2,2+i+1)
	plot(allauto['wak'][neurons[K==i]].loc[-200:200], color = 'grey', alpha = 0.5)


figure()
for i in range(2):
	plot(allauto['wak'][neurons[K==i]].mean(1).loc[-500:500], color = 'grey', alpha = 0.5)


figure()
count = 1
for j in range(len(np.unique(K))):
	for e in allauto.keys():	
		subplot(len(np.unique(K)),3,count)
		tmp = allauto[e][neurons[K==j]]
		plot(tmp.mean(1))
		count += 1
		title(e)
show()

figure()
count = 1
for e in allauto.keys():	
	subplot(1,3,count)
	for j in range(len(np.unique(K))):		
		tmp = allauto[e][neurons[K==j]]
		plot(tmp.mean(1))
	count += 1
	title(e)
show()

figure()
for i, n in enumerate(neurons[K==0]):
	subplot(7,7,i+1)
	plot(allauto['wak'][n].loc[-200:200])

figure()
for i, n in enumerate(neurons[K==1]):
	subplot(7,5,i+1)
	plot(allauto['wak'][n].loc[-200:200])

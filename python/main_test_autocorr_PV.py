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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

datasets = [	'LMN-ADN/A5002/A5002-200303A',
				'LMN-ADN/A5002/A5002-200303B',
       			'LMN-ADN/A5002/A5002-200304A', 
       			'LMN-ADN/A5002/A5002-200305A',
       			'LMN-ADN/A5002/A5002-200309A',
       			'LMN/A1407/A1407-190416', 
       			'LMN/A1407/A1407-190417',
       			'LMN/A1407/A1407-190422']

allauto = {e:[] for e in ['wak', 'rem', 'sws']}
allfrates = {e:[] for e in ['wak', 'rem', 'sws']}
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
	sws_ep								= loadEpoch(path, 'sws')
	rem_ep								= loadEpoch(path, 'rem')

	# Only taking the first wake ep
	wake_ep = wake_ep.loc[[0]]

	# # Taking only neurons from LMN
	if 'A50' in s:
		# spikes = {n:spikes[n] for n in np.intersect1d(np.where(shank.flatten()>2)[0], np.where(shank.flatten()<4)[0])}
		spikes = {n:spikes[n] for n in np.where(shank.flatten()==3)[0]}
	if 'A14' in s:
		spikes = {n:spikes[n] for n in np.where(shank.flatten()==1)[0]}
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
	
	# spikes = {n:spikes[n] for n in tokeep}
	neurons = [name+'_'+str(n) for n in spikes.keys()]
	hd_index.append([n for n in neurons if int(n.split('_')[1]) in tokeep])

	############################################################################################### 
	# COMPUTE AUTOCORRS
	###############################################################################################
	for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):
		autocorr, frates = compute_AutoCorrs(spikes, ep, 0.5, 200)
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
	auto = auto.rolling(window = 40, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
	auto = auto.loc[0.5:40]
	# 5 firing rate > 1 
	auto  = auto[frates[np.sum(frates>1.0,1)==3].index]
	# Drop nan
	auto = auto.dropna(1)
	halfauto[e] = auto
	allindex.append(auto.columns)

hd_index = np.hstack(hd_index)
neurons = np.intersect1d(np.intersect1d(allindex[0], allindex[1]), allindex[2])
# neurons = np.intersect1d(neurons, fr_index)

# shanks_index = pd.concat(shanks_index)

data = np.hstack([halfauto[e][neurons].values.T for e in halfauto.keys()])

from umap import UMAP
from sklearn.cluster import KMeans

hd_index = np.intersect1d(neurons, hd_index)
c = pd.Series(index = neurons, data = 0)
# c.loc[hd_index] = 1


X = TSNE(2, 10).fit_transform(data)

U = UMAP(n_neighbors = 20, n_components = 2).fit_transform(data)

K = KMeans(n_clusters = 2, random_state = 0).fit(U).labels_

U = pd.DataFrame(index = neurons, data = X)

figure()

scatter(U[0], U[1], s = 50, c = K)
scatter(U.loc[hd_index,0], U.loc[hd_index,1], s = 20, c = 'red', label = 'HD')
legend()


colors = ['green', 'orange']

figure()
count = 1
for j in range(2):
	for e in allauto.keys():	
		subplot(3,3,count)
		plot(allauto[e][neurons[K==j]], color = colors[j])
		title(e)
		count += 1
for e in allauto.keys():
	subplot(3,3,count)
	plot(allauto[e][neurons[K==0]].mean(1), color = 'green', linewidth = 2)
	# plot(allauto[e][neurons[K==0]], color ='red', alpha = 0.3)
	plot(allauto[e][neurons[K==1]].mean(1), color = 'orange', linewidth = 2)
	# plot(allauto[e][neurons[K==1]], color ='green', alpha = 0.3)
	title(e)
	count += 1



show()
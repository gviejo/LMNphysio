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

allwave = []
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
	# 
	###############################################################################################
	meanwavef, maxch = loadMeanWaveforms(path)	
	meanwavef = meanwavef[list(spikes.keys())]
	for i,n in zip(meanwavef.columns,neurons):
		tmp = meanwavef[i].values.reshape(40,len(meanwavef[i].values)//40)
		allwave.append(pd.DataFrame(data = tmp[:,maxch[i]], columns = [n]))
	
	
allwave = pd.concat(allwave, 1) 

sys.exit()


neurons = allwave.columns

hd_index = np.hstack(hd_index)

data = allwave.values.T

from umap import UMAP
from sklearn.cluster import KMeans


U = UMAP(n_neighbors = 20, n_components = 2).fit_transform(data)

K = KMeans(n_clusters = 2, random_state = 0).fit(U).labels_

U = pd.DataFrame(index = neurons, data = U)

figure()

scatter(U[0], U[1], s = 50, c = K)
scatter(U.loc[hd_index,0], U.loc[hd_index,1], s = 20, c = 'red', label = 'HD')
legend()


colors = ['green', 'orange']

figure()
count = 1
for j in range(2):
	subplot(1,2,count)
	plot(allwave[neurons[K==j]], color = colors[j])
	count += 1


neurons2 = neurons[K==0]
hd_index2 = np.intersect1d(hd_index, neurons2)
allwave2 = allwave[neurons2]
data2 = allwave2.values.T

U2 = UMAP(n_neighbors = 10, n_components = 2).fit_transform(data2)
K2 = KMeans(n_clusters = 2, random_state = 0).fit(U2).labels_
U2 = pd.DataFrame(index = neurons2, data = U2)

figure()

scatter(U2[0], U2[1], s = 50, c = K2)
scatter(U2.loc[hd_index2,0], U2.loc[hd_index2,1], s = 20, c = 'red', label = 'HD')
legend()


colors = ['green', 'orange']

figure()
count = 1
for j in range(2):
	subplot(2,2,count)
	plot(allwave[neurons2[K2==j]], color = colors[j])
	count += 1
subplot(2,2,3)
plot(allwave[neurons2[K2==0]], color = colors[0], alpha = 0.3)
plot(allwave[neurons2[K2==1]], color = colors[1], alpha = 0.3)

show()
subplot(2,2,4)
plot(allwave[neurons2[K2==0]].mean(1), color = colors[0])
plot(allwave[neurons2[K2==1]].mean(1), color = colors[1])

show()
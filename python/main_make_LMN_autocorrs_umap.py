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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_KS25.txt'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)


allauto = {e:[] for e in ['wak', 'rem', 'sws']}
allfrates = {e:[] for e in ['wak', 'rem', 'sws']}
hd_index = []
shanks_index = []
alltcurves = []
allvcurves = []
allscurves = []



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
	if 'A5002' in s:
		spikes = {n:spikes[n] for n in np.where(shank==3)[0]}

	if 'A5011' in s:
		spikes = {n:spikes[n] for n in np.where(shank==5)[0]}
	
	

	
	# ############################################################################################### 
	# # COMPUTING TUNING CURVES
	# ###############################################################################################
	tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)	
	tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

	velo_curves = computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=False)
	speed_curves = computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)

	velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
	speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


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

	# Checking firing rate
	spikes = {n:spikes[n] for n in tokeep}
	mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])		
	tokeep = mean_frate[mean_frate.loc[tokeep,'sws']>2].index.values

	spikes = {n:spikes[n] for n in tokeep}
	neurons = [name+'_'+str(n) for n in spikes.keys()]
	hd_index.append([n for n in neurons if int(n.split('_')[1]) in tokeep])

	tcurves = tuning_curves[tokeep]
	tcurves.columns = neurons
	velo_curves = velo_curves[tokeep]
	velo_curves.columns = neurons
	speed_curves = speed_curves[tokeep]
	speed_curves.columns = neurons
	
	############################################################################################### 
	# COMPUTE AUTOCORRS
	###############################################################################################
	for e, ep, tb, tw in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep], [1, 1, 1], [2000, 2000, 2000]):
		autocorr, frates = compute_AutoCorrs(spikes, ep, tb, tw)
		autocorr.columns = pd.Index(neurons)
		frates.index = pd.Index(neurons)
		allauto[e].append(autocorr)
		allfrates[e].append(frates)

	#######################
	# SAVING
	#######################
	alltcurves.append(tcurves)	
	allvcurves.append(velo_curves)
	allscurves.append(speed_curves)


for e in allauto.keys():
	allauto[e] = pd.concat(allauto[e], 1)
	allfrates[e] = pd.concat(allfrates[e])

alltcurves = pd.concat(alltcurves, 1)
allvcurves = pd.concat(allvcurves, 1)
allscurves = pd.concat(allscurves, 1)

frates = pd.DataFrame.from_dict(allfrates)
hd_index = np.hstack(hd_index)


allindex = []
halfauto = {}
for e in allauto.keys():
	# 1. starting at 2
	auto = allauto[e].loc[0:]
	# auto = allauto[e]
	# 3. lower than 100 
	auto = auto.drop(auto.columns[auto.apply(lambda col: col.max() > 20.0)], axis = 1)
	# # 4. gauss filt
	auto = auto.rolling(window = 10, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
	auto = auto.loc[0:40]
	# Drop nan
	auto = auto.dropna(1)
	halfauto[e] = auto
	allindex.append(auto.columns)



neurons = np.intersect1d(np.intersect1d(allindex[0], allindex[1]), allindex[2])
# neurons = np.intersect1d(neurons, fr_index)

# shanks_index = pd.concat(shanks_index)

# data = np.hstack([halfauto[e][neurons].values.T for e in halfauto.keys()])
# data = halfauto['wak'].values.T
data = np.hstack([halfauto[e][neurons].values.T for e in ['wak', 'sws']])

from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec

U = UMAP(n_neighbors = 10, n_components = 2).fit_transform(data)
K = KMeans(n_clusters = 2, random_state = 0).fit(U).labels_


figure()
scatter(U[:,0], U[:,1], c = K)

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
gs = GridSpec(1,3)
tc = []
for i in range(2):
	subplot(gs[0,i])
	tmp = alltcurves[neurons[K==i]]
	tmp2 = centerTuningCurves(tmp)
	tmp2 = tmp2/tmp2.loc[0]
	plot(tmp2, color = 'grey')
	plot(tmp2.mean(1))
	tc.append(tmp2.mean(1))
subplot(gs[0,-1])
plot(tc[0])
plot(tc[1])

# ahv = []
# for i in range(2):
# 	subplot(gs[1,i])
# 	tmp = allvcurves[neurons[K==i]]		
# 	plot(tmp, color = 'grey')
# 	ahv.append(tmp.mean(1))
# subplot(gs[1,-1])
# plot(ahv[0])
# plot(ahv[1])
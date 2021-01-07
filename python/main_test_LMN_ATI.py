import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from matplotlib.colors import hsv_to_rgb
import hsluv
from pycircstat.descriptive import mean as circmean
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from umap import UMAP
from matplotlib import gridspec
from itertools import product

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = list(np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)


mapping 		= []
tcurves 		= []
ahvcurves 		= []
peaks 			= []
zvalues 		= []
zvalues_s 		= []
zvalues_a 		= []
tc_jittered = {}

for s in datasets:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)	
	if 'A5002-200305A' in s:
		wake_ep = wake_ep.loc[[1]]
	else:
		wake_ep = wake_ep.loc[[0]]

	sws_ep			= loadEpoch(path, 'sws')
	rem_ep			= loadEpoch(path, 'rem')
	
	meanwavef, maxch = loadMeanWaveforms(path)

	# TO RESTRICT BY SHANK
	if 'A5002' in s:
		spikes 			= {n:spikes[n] for n in np.where(shank>2)[0]}

	meanwavef 		= meanwavef[list(spikes.keys())]
	neurons 		= [name+'_'+str(n) for n in spikes.keys()]
	meanwavef.columns = pd.Index(neurons)

	speed = computeSpeed(position[['x', 'z']], wake_ep)
	speed = speed.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)
	idx = np.diff((speed > 0.005)*1.0)
	start = np.where(idx == 1)[0]
	end = np.where(idx == -1)[0]
	if start[0] > end[0]:
		start = np.hstack(([0], start))
	if start[-1] > end[-1]:
		end = np.hstack((end, [len(idx)]))

	newwake_ep = nts.IntervalSet(start = speed.index.values[start], end = speed.index.values[end])
	newwake_ep = newwake_ep.drop_short_intervals(1, time_units='s')


	######################
	# TUNING CURVEs
	######################
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

	neurons = [name+'_'+str(n) for n in spikes.keys()]

	tcurve 			= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)	
	tcurve 			= smoothAngularTuningCurves(tcurve, 20, 2)	
	tcurve.columns	= pd.Index(neurons)
	peak 			= pd.Series(index=tcurve.columns,data = np.array([circmean(tcurve.index.values, tcurve[i].values) for i in tcurve.columns]))
	hd_neurons 		= [name+'_'+str(n) for n in tokeep]
	tcurve 			= tcurve[hd_neurons]
	peak 			= peak.loc[hd_neurons]
	
	
	#####################
	# SHIFTING ANGLE IN TIME
	#####################
	bin_size = 20
	zvalue = []
	for n in neurons:
		tc_jittered[n] = []

	bins = np.arange(-2000, 2000, bin_size)

	for i, t in enumerate(bins):
		hd_spikes = {}
		for j in tokeep:
			tmp 			= spikes[j].index.values + t*1000
			hd_spikes[j] 	= nts.Ts(t = tmp)
		
		tc 				= computeAngularTuningCurves(hd_spikes, position['ry'], newwake_ep, 61)	
		tc 				= smoothAngularTuningCurves(tc, 20, 2)		
		spatialinfo 	= computeSpatialInfo(tc, position['ry'], newwake_ep)		
		spatialinfo.index = hd_neurons
		zvalue.append(spatialinfo.loc[hd_neurons,'SI'].values)
		for j, n in zip(tc.columns, hd_neurons):
			tc_jittered[n].append(tc[j])

	zvalue = np.array(zvalue)
	zvalue = pd.DataFrame(index = bins,
							data = zvalue, 
							columns = hd_neurons)
	for n in hd_neurons:
		tc_jittered[n] = pd.concat(tc_jittered[n], 1)
		tc_jittered[n].columns = bins


	# figure()
	# for i in range(20):
	# 	subplot(4,5,i+1)
	# 	plot(zvalue.iloc[:,i])

	# figure()
	# for i,n in zip(range(20), tokeep):
	# 	subplot(4,5,i+1)	
	# 	imshow(tc_jittered[n], aspect = 'auto', cmap = 'jet')



	######################
	# TOSAVE
	######################
	tcurves.append(tcurve)
	peaks.append(peak)
	# mapping.append(mapp)
	zvalues.append(zvalue)
	# zvalues_s.append(zvalue_s)
	# zvalues_a.append(zvalue_a)


tcurves 		= pd.concat(tcurves, 1)
peaks 			= pd.concat(peaks)
# mapping 		= pd.concat(mapping)
zvalues 		= pd.concat(zvalues, 1)
# zvalues_s 		= pd.concat(zvalues_s, 1)
# zvalues_a 		= pd.concat(zvalues_a, 1)



figure()
tmp = zvalues
a = tmp - tmp.min(0)
a = a / a.max(0)
plot(a, color = 'grey')
plot(a.mean(1), color = 'red')


zvalues = zvalues.astype(np.float32)

peak = zvalues.idxmax(0)

bins = [-1000,0,1000]
groups = peak.groupby(np.digitize(peak.values, bins)).groups

bins = tc_jittered[groups[2][0]].columns.values


figure()
subplot(121)
tmp = zvalues[groups[2]]
a = tmp - tmp.min(0)
a = a / a.max(0)
plot(a, color = 'grey')
plot(a.mean(1), color = 'red')
axvline(a.mean(1).idxmax())
subplot(122)
tmp = zvalues[groups[2]].loc[0:200]
a = tmp - tmp.min(0)
a = a / a.max(0)
plot(a, color = 'grey')
plot(a.mean(1), color = 'red')



figure()
for i in range(len(groups[2])):
	subplot(5,5,i+1)
	tmp = tc_jittered[groups[2][i]]
	imshow(tmp, aspect = 'auto')
	xticks(np.arange(0, len(bins), 50), bins[np.arange(0, len(bins), 50)])
	axvline(np.where(bins == peak[groups[2][i]])[0])

figure()
for i in range(len(groups[2])):
	subplot(5,5,i+1, projection='polar')
	n = groups[2][i]
	plot(tc_jittered[n][peak[n]])
	plot(tc_jittered[n][0], '--')



figure()
for i in range(len(groups[2])):
	subplot(6,5,i+1, projection='polar')
	n = groups[2][i]
	plot(tc_jittered[n][peak[n]])
	plot(tc_jittered[n][0], '--')

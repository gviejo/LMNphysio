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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

# A5002
datasets = ['LMN-ADN/A5002/'+s for s in infos['A5002'].index[1:-1]]
datasets.remove('LMN-ADN/A5002/A5002-200306A')

# A1407
# datasets = ['LMN/A1407/'+s for s in infos['A1407'].index[10:-2]]
# datasets.remove('LMN/A1407/A1407-190406')


mapping 		= []
ccall 			= []
tcurves 		= []
ahvcurves 		= []
cc_sync 		= []
cc_async 		= []
ahv_sync 		= []
ahv_async 		= []
tcurves_sync 	= []
tcurves_async 	= []
peaks 			= []


for s in ['LMN-ADN/A5002/A5002-200304A']:
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
		spikes 			= {n:spikes[n] for n in np.where(shank==3)[0]}

	meanwavef 		= meanwavef[list(spikes.keys())]
	neurons 		= [name+'_'+str(n) for n in spikes.keys()]
	meanwavef.columns = pd.Index(neurons)

	######################
	# TUNING CURVEs
	######################
	tcurve 			= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)	
	tcurve 			= smoothAngularTuningCurves(tcurve, 20, 2)
	tokeep, stat 	= findHDCells(tcurve, z = 10, p = 0.001)
	tcurve.columns	= pd.Index(neurons)
	peak 			= pd.Series(index=tcurve.columns,data = np.array([circmean(tcurve.index.values, tcurve[i].values) for i in tcurve.columns]))
	hd_neurons 		= [name+'_'+str(n) for n in tokeep]	
	tcurve 			= tcurve[hd_neurons]
	peak 			= peak.loc[hd_neurons]	
	pairs 			= list(product(hd_neurons, hd_neurons))

	######################
	# DECIMATED TUNING CURVES
	######################
	tcurve_s 	= pd.DataFrame(columns = pairs)
	tcurve_a 	= pd.DataFrame(columns = pairs)
	
	for p in pairs:
		print(p)
		spk1 = spikes[int(p[0].split("_")[1])].restrict(wake_ep).as_units('ms').index.values
		spk2 = spikes[int(p[1].split("_")[1])].restrict(wake_ep).as_units('ms').index.values
		spksync = []
		spkasync = []
		for t in spk2:			
			if np.sum(np.abs(t-spk1)<=5):
				spksync.append(t)
			else:
				spkasync.append(t)
		
		spksync = nts.Ts(t = np.array(spksync), time_units = 'ms')
		spkasync = nts.Ts(t = np.array(spkasync), time_units = 'ms')
		dec_spikes = {0:spksync,1:spkasync}
		tc = computeAngularTuningCurves(dec_spikes, position['ry'], wake_ep, 61)
		tcurve_s[p] = tc[0]
		tcurve_a[p] = tc[1]
	
	#############################
	# SYNC SPIKES
	#############################
	sws_ep = sws_ep.loc[[41]].reset_index(drop=True)
	spikes_sync = {}
	for p in pairs:
		spk1 = spikes[int(p[0].split("_")[1])].restrict(sws_ep).as_units('ms').index.values
		spk2 = spikes[int(p[1].split("_")[1])].restrict(sws_ep).as_units('ms').index.values			
		tmp = []
		for t in spk2:
			if np.sum(np.abs(t-spk1)<=5):
				tmp.append(t)
		if len(tmp):
			spikes_sync[p] = nts.Ts(t = np.array(tmp), time_units = 'ms')

	
	##########################
	# SLEEP DECODING
	##########################	
	tcs2 = tcurve_s[list(spikes_sync.keys())]
	decoded, proba_angle, spike_counts = decodeHD(tcs2, spikes_sync, sws_ep, bin_size = 20, px = None)


	######################
	# TOSAVE
	######################
	tcurves.append(tcurve)	
	ahvcurves.append(ahvcurve)
	ccall.append(cc)
	cc_sync.append(cc_s)
	cc_async.append(cc_a)
	ahv_sync.append(ahv_s)
	ahv_async.append(ahv_a)
	tcurves_sync.append(tcurve_s)
	tcurves_async.append(tcurve_a)
	peaks.append(peak)
	

tcurves 		= pd.concat(tcurves, 1)
ahvcurves 		= pd.concat(ahvcurves, 1)
ccall 			= pd.concat(ccall, 1)
cc_sync 		= pd.concat(cc_sync, 1)
cc_async		= pd.concat(cc_async, 1)
ahv_sync 		= pd.concat(ahv_sync, 1)
ahv_async 		= pd.concat(ahv_async, 1)
tcurves_sync 	= pd.concat(tcurves_sync, 1)
tcurves_async 	= pd.concat(tcurves_async, 1)
peaks 			= pd.concat(peaks)

diffs = pd.Series(index = ahv_sync.columns, data = [peaks.loc[p[0]]-peaks.loc[p[1]] for p in ahv_sync.columns])
diffs[diffs>np.pi] -= 2*np.pi
diffs[diffs<-np.pi] += 2*np.pi

H = (diffs+np.pi)/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)


dtcurves = []
for p in tcurves_sync.columns:
	# tmp = tcurves[p[1]] - tcurves_sync[p]
	x = tcurves[p[1]].index.values - tcurves[p[1]].index[tcurves[p[1]].index.get_loc(peaks[p[1]], method='nearest')]
	x[x<-np.pi] += 2*np.pi
	x[x>np.pi] -= 2*np.pi
	tmp = pd.Series(index = x, data = tcurves_sync[p].values).sort_index()
	dtcurves.append(tmp.values)
dtcurves = pd.DataFrame(index = np.linspace(-np.pi, np.pi, tcurve.shape[0]+1)[0:-1], data = np.array(dtcurves).T, columns = tcurves_sync.columns)

sys.exit()


tcurves_ = tcurves[[p[1] for p in tcurves_sync.columns]]
tcurves_.columns = tcurves_sync.columns
tcurves2 = centerTuningCurves(tcurves_)
tcurves3 = offsetTuningCurves(tcurves2, diffs)


tcurves_sync2 = centerTuningCurves(tcurves_sync)
tcurves_sync3 = offsetTuningCurves(tcurves_sync2, diffs)

tcurves_async2 = centerTuningCurves(tcurves_async)
tcurves_async3 = offsetTuningCurves(tcurves_async2, diffs)



figure()
subplot(231)
for p in tcurves_sync3.columns:
	plot(tcurves_sync3[p], color = hsv_to_rgb([H[p], 1,1]))
subplot(232, projection = 'polar')
for p in tcurves_sync3.columns:
	plot(tcurves_sync3[p], color = hsv_to_rgb([H[p], 1,1]))
subplot(233)
plot(tcurves_sync3.mean(1))
subplot(234)
for p in tcurves_async3.columns:
	plot(tcurves_async3[p], color = hsv_to_rgb([H[p], 1,1]))
subplot(235, projection = 'polar')
for p in tcurves_async3.columns:
	plot(tcurves_async3[p], color = hsv_to_rgb([H[p], 1,1]))
subplot(236)
plot(tcurves_async3.mean(1))
plot(tcurves3.mean(1))


###########
# SYNC TCURVES AS A FUNCTION OF DIFF
###########
n = 3
idxs = np.array_split(diffs.abs().sort_values().index.values, n)
gs = gridspec.GridSpec(2,n)
for i, idx in enumerate(idxs):
	subplot(gs[0,i],projection = 'polar')
	for p in idx:
		plot(tcurves_sync[p], color = hsv_to_rgb([H[p], 1,1]))
	subplot(gs[1,i])
	for p in idx:
		plot(tcurves_sync[p], color = hsv_to_rgb([H[p], 1,1]))


figure()
subplot(121, projection = 'polar')
for p in tcurves_sync.columns:
	plot(tcurves_sync[p]/tcurves_sync[p].max(), color = hsv_to_rgb([H[p], 1,1]))
subplot(122, projection = 'polar')
for p in tcurves_async.columns:
	plot(tcurves_async[p], color = hsv_to_rgb([H[p], 1,1]))






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


# for s in datasets:
for s in np.array(datasets)[[6,7,8]]:
# for s in np.array(datasets)[[6,7,8,9,10,11]]:
# for s in ['LMN-ADN/A5002/A5002-200304A']:
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

	######################
	# AHV CURVES
	######################
	ahvcurve 		= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 61, norm=False)
	ahvcurve 		= ahvcurve.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1)
	ahvcurve.columns = pd.Index(neurons)
	ahvcurve 		= ahvcurve[hd_neurons]

	######################
	# NORMAL CROSS-CORR
	######################
	spks = spikes
	binsize = 0.5
	nbins = 100
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

	cc = pd.DataFrame(index = times, columns = list(product(hd_neurons, hd_neurons)))	
	
	for i,j in cc.columns:
		if i != j:
			spk1 = spks[int(i.split("_")[1])].restrict(wake_ep).as_units('ms').index.values
			spk2 = spks[int(j.split("_")[1])].restrict(wake_ep).as_units('ms').index.values		
			tmp = crossCorr(spk1, spk2, binsize, nbins)					
			cc[(i,j)] = tmp			
	cc = cc.dropna(1)	

	# ccfast = cc.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 2)
	# ccslow = cc.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 20)
	# ccmod = (cc - ccslow)/cc.std(0)
	# ccmod2 = (ccfast - ccslow)/ccfast.std(0)

	######################
	# SYNC/ASYNC TUNING CURVES
	######################
	spks = [spikes[int(n.split('_')[1])].restrict(wake_ep).as_units('ms').index.values for n in hd_neurons]
	spksync, spkasync = getSpikesSyncAsync(spks, hd_neurons)	
	
	tcurve_s = computeAngularTuningCurves(spksync, position['ry'], wake_ep, 61)
	tcurve_a = computeAngularTuningCurves(spkasync, position['ry'], wake_ep, 61)
	ahv_s = computeAngularVelocityTuningCurves(spksync, position['ry'], wake_ep, nb_bins = 31, norm=True)
	ahv_a = computeAngularVelocityTuningCurves(spkasync, position['ry'], wake_ep, nb_bins = 31, norm=True)


	##########################
	# SAMPLING SPIKES FROM ANGULAR TUNING CURVES
	##########################		
	spikes_random = sampleSpikesFromAngularPosition(tcurve, position['ry'], wake_ep)

	######################
	# RANDOM SYNC/ASYNC TUNING CURVES
	######################
	tcurve_r_s 	= pd.DataFrame(columns = cc.columns)
	tcurve_r_a 	= pd.DataFrame(columns = cc.columns)

	for n in spikes_random.keys():
		for m in spikes_random.keys():
			if m != n:
				spksync = {}
				spkasync = {}
				for i in range(len(spikes_random[n])):
					spks = [spikes[int(m.split('_')[1])].restrict(wake_ep).as_units('ms').index.values,
							spikes_random[n][i].restrict(wake_ep).as_units('ms').index.values]
					tmp1, tmp2 = getSpikesSyncAsync(spks, [m,n])
					spksync[i] = tmp1[(m,n)]
					spkasync[i] = tmp2[(m,n)]

				tc_s = computeAngularTuningCurves(spksync, position['ry'], wake_ep, 61)
				tc_a = computeAngularTuningCurves(spkasync, position['ry'], wake_ep, 61)
				tcurve_r_s[(m,n)] = tc_s.mean(1)
				tcurve_r_a[(m,n)] = tc_a.mean(1)

	##########################
	# SAMPLING SPIKES FROM AHV TUNING CURVES
	##########################		
	spikes_random = sampleSpikesFromAngularVelocity(ahvcurve, position['ry'], wake_ep)
	
	######################
	# RANDOM SYNC/ASYNC TUNING CURVES
	######################
	ahv_r_s 	= pd.DataFrame(columns = cc.columns)
	ahv_r_a 	= pd.DataFrame(columns = cc.columns)

	for n in spikes_random.keys():
		for m in spikes_random.keys():
			if m != n:
				spksync = {}
				spkasync = {}
				for i in range(len(spikes_random[n])):
					spks = [spikes[int(m.split('_')[1])].restrict(wake_ep).as_units('ms').index.values,
							spikes_random[n][i].restrict(wake_ep).as_units('ms').index.values]
					tmp1, tmp2 = getSpikesSyncAsync(spks, [m,n])
					spksync[i] = tmp1[(m,n)]
					spkasync[i] = tmp2[(m,n)]

				av_s = computeAngularVelocityTuningCurves(spksync, position['ry'], wake_ep, 61, norm=False)
				av_a = computeAngularVelocityTuningCurves(spkasync, position['ry'], wake_ep, 61, norm=False)
				ahv_r_s[(m,n)] = av_s.mean(1)
				ahv_r_a[(m,n)] = av_a.mean(1)




	#############################

	# diffs = pd.Series(index=cc.columns, data = [peak.loc[p[0]] - peak.loc[p[1]] for p in cc.columns])
	# diffs[diffs>np.pi] -= 2*np.pi
	# diffs[diffs<-np.pi] += 2*np.pi

	# a = (tcurve_s - tcurve_r_s)/tcurve_r_s

	# plot(diffs.values, a.mean(0).values, 'o')
	


	# for p in idx:
	# 	gs = gridspec.GridSpec(2,4)
	# 	figure()
	# 	subplot(gs[0,0])
	# 	plot(cc[p])
	# 	title(p)
	# 	subplot(gs[0,1], projection = 'polar')
	# 	plot(tcurve[list(p)])
	# 	subplot(gs[0,2], projection = 'polar')
	# 	plot(tcurve_a[p], '-', label='async', color = 'red')		
	# 	plot(tcurve_r_a[p], '--', label='random async', color = 'red')
	# 	legend()
	# 	subplot(gs[0,3], projection = 'polar')
	# 	plot(tcurve_r_s[p], '--', label='random sync', color='green')
	# 	plot(tcurve_s[p], '-', label='sync', color='green')
	# 	legend()
	# 	subplot(gs[1,0])
	# 	for i,j,c in zip(p,range(2),['red', 'blue']):
	# 		tmp = meanwavef[i].values.reshape(40,16)
	# 		plot(np.arange(0+j*40,40+j*40),tmp+np.arange(16)*200, color = c)		
	# 	# subplot(gs[1,1])
	# 	# plot(ahvcurve[list(p)])
	# 	# subplot(gs[1,2])
	# 	# plot(ahv_a[p], '-', label='async', color = 'red')		
	# 	# plot(ahv_r_a[p], '--', label='random async', color = 'red')
	# 	# legend()
	# 	# subplot(gs[1,3])
	# 	# plot(ahv_r_s[p], '--', label='random sync', color='green')
	# 	# plot(ahv_s[p], '-', label='sync', color='green')
	# 	# legend()		
	# 	show(block=True)



	# plot(tcurve[p[1]]/tcurve[p[1]].max())
	# plot(tcurve_s[p]/tcurve_s[p].max(), color = 'green', label = 'sync')
	# plot(tcurve_a[p]/tcurve_a[p].max(), color = 'red', label = 'async')
	# plot(tcurve_r_s[p]/tcurve_r_s[p].max(), '--', color = 'green', label = 'sync')
	# plot(tcurve_r_a[p]/tcurve_r_a[p].max(), '--', color = 'red', label = 'async')


	######################
	# TOSAVE
	######################
	tcurves.append(tcurve)	
	ahvcurves.append(ahvcurve)
	ccall.append(cc)
	# cc_sync.append(cc_s)
	# cc_async.append(cc_a)
	ahv_sync.append(ahv_s)
	ahv_async.append(ahv_a)
	tcurves_sync.append(tcurve_s)
	tcurves_async.append(tcurve_a)
	peaks.append(peak)
	

tcurves 		= pd.concat(tcurves, 1)
ahvcurves 		= pd.concat(ahvcurves, 1)
ccall 			= pd.concat(ccall, 1)
# cc_sync 		= pd.concat(cc_sync, 1)
# cc_async		= pd.concat(cc_async, 1)
ahv_sync 		= pd.concat(ahv_sync, 1)
ahv_async 		= pd.concat(ahv_async, 1)
tcurves_sync 	= pd.concat(tcurves_sync, 1)
tcurves_async 	= pd.concat(tcurves_async, 1)
peaks 			= pd.concat(peaks)


sys.exit()
#################
# SMOOTHING AHV
#################
ahv_sync = ahv_sync.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1)
ahv_async = ahv_async.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1)


ahv2 = pd.concat([ahvcurves[p[1]] for p in ahv_sync.columns], 1)
ahv2.columns = ahv_sync.columns

diffs = pd.Series(index = ahv_sync.columns, data = [peaks.loc[p[0]]-peaks.loc[p[1]] for p in ahv_sync.columns])
diffs[diffs>np.pi] -= 2*np.pi
diffs[diffs<-np.pi] += 2*np.pi

dahv_sync = ahv_sync - ahv2
dahv_async = ahv_async - ahv2

tmp = np.vstack((dahv_sync.values, dahv_async.values))

H = (diffs+np.pi)/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)



ump = UMAP(n_neighbors = 50, min_dist = 1e-6).fit_transform(tmp.T)
# kmeans = KMeans(n_clusters = 5).fit(ump)
# labels = kmeans.labels_
clustering = SpectralClustering(n_clusters = 3).fit(ump)
labels = clustering.labels_

# scatter(ump[:,0], ump[:,1], c = labels)

figure()
gs = gridspec.GridSpec(4,1+len(np.unique(labels)))
subplot(gs[0,0])
scatter(ump[:,0], ump[:,1], c = RGB)
subplot(gs[1,0])
scatter(ump[:,0], ump[:,1], c = labels)
for i in range(len(np.unique(labels))):
	subplot(gs[0,i+1])
	plot(dahv_sync.iloc[:,labels==i], color = 'grey', alpha = 0.5, linewidth = 1)
	plot(dahv_sync.iloc[:,labels==i].mean(1), color = 'green', linewidth = 3)
	subplot(gs[1,i+1])
	plot(dahv_async.iloc[:,labels==i], color = 'grey', alpha = 0.5, linewidth = 1)
	plot(dahv_async.iloc[:,labels==i].mean(1), color = 'red', linewidth = 3)
	subplot(gs[2,i+1])
	plot(dahv_sync.iloc[:,labels==i].mean(1), color = 'green', linewidth = 3)
	plot(dahv_async.iloc[:,labels==i].mean(1), color = 'red', linewidth = 3)
	subplot(gs[3,i+1], projection = 'polar')
	hist(diffs.loc[dahv_sync.columns.values[labels==i]], 30)

subplot(gs[2,0])
tmp2 = ahv_sync.loc[-2:2][diffs.abs().sort_values().index].values.T
imshow(scipy.ndimage.gaussian_filter(tmp2, 2), aspect = 'auto')
subplot(gs[3,0])
tmp3 = ahv_async.loc[-2:2][diffs.abs().sort_values().index].values.T
imshow(scipy.ndimage.gaussian_filter(tmp3, 2), aspect = 'auto')

sys.exit()

	# ccmod2 = ccmod2[ccmod2.loc[-3:3].mean().sort_values().index]




# p = ccmod2.columns[-4]
p = ccmod2.columns[-9]



# p = ('A5002-200304A_67', 'A5002-200304A_70')
for p in ccmod2.columns[::-1]:
# for p in [('A5002-200304A_67', 'A5002-200304A_81')]:
	print(p)
	# COMPUTE TUNING CURVES WITHOUT SYNCHRONE SPIKES
	spk1 = spks[int(p[0].split("_")[1])].restrict(wake_ep).as_units('ms').index.values
	spk2 = spks[int(p[1].split("_")[1])].restrict(wake_ep).as_units('ms').index.values
	spk3 = []
	spk4 = []
	for t in spk2:
		if np.sum(np.abs(t-spk1)<3):
			spk3.append(t)
		else:
			spk4.append(t)
	cc_less = crossCorr(spk1, np.array(spk4), binsize, nbins)
	cc_less = (cc_less - ccslow[p])/cc_less
	spk3 = nts.Ts(t = np.array(spk3), time_units = 'ms')
	spk4 = nts.Ts(t = np.array(spk4), time_units = 'ms')
	dec_spikes = {0:spk3,1:spk4}
	tcurves2 = computeAngularTuningCurves(dec_spikes, position['ry'], wake_ep, 61)
	ahvcurves2 		= computeAngularVelocityTuningCurves(dec_spikes, position['ry'], wake_ep, nb_bins = 30, norm=True)
	
	figure(figsize=(10,6))
	subplot(231)
	plot(ccmod[p])
	plot(ccmod2[p])
	plot(cc_less)
	subplot(232, projection = 'polar')
	plot(tcurves[list(p)])
	plot(tcurves2[0], '--', label = 'synchrone')
	plot(tcurves2[1], '--', label = 'asynchrone')
	legend()
	subplot(233)
	plot(tcurves[list(p)])
	plot(tcurves2[0], '--', label = 'synchrone')
	plot(tcurves2[1], '--', label = 'asynchrone')
	legend()
	subplot(234)
	for i,j,c in zip(p,range(2),['red', 'blue']):
		tmp = meanwavef[i].values.reshape(40,16)
		plot(np.arange(0+j*40,40+j*40),tmp+np.arange(16)*200, color = c)
	subplot(235)
	plot(ahvcurves[list(p)])
	plot(ahvcurves2[0], '--', label = 'synchrone')
	plot(ahvcurves2[1], '--', label = 'asynchrone')		
	subplot(236, projection = 'polar')
	tmp = tcurves[list(p)]
	plot(tmp/tmp.max())
	plot(tcurves2[0]/tcurves2[0].max(), '--', label = 'synchrone')
	plot(tcurves2[1]/tcurves2[1].max(), '--', label = 'asynchrone')

	show(block=True)

sys.exit()

ccall.append(cc)

	# figure()
	# gs = gridspec.GridSpec(2,4)
	# for i,j,p in zip([0,0,1,1],[2,3,2,3],np.array_split(ccmod2.columns.values, 4)):
	# 	subplot(gs[i,j])
	# 	plot(ccmod2[p])
	# 	ylim(-4,4)
	# subplot(gs[:,0:2])
	# imshow(ccmod2.values.T)


	# figure()
	# gs = gridspec.GridSpec(2,4)
	# for i,j in enumerate([2,12,14,21]):
	# 	p = ccmod.columns[j]
	# 	subplot(gs[0,i], projection = 'polar')
	# 	tmp = tcurves[list(p)]
	# 	plot(tmp/tmp.max())
	# 	subplot(gs[1,i])
	# 	plot(ccmod[p], label = p)
	# 	plot(ccmod2[p])
	# 	legend()



# SYNAPTIC DETECTION
ccfast = ccall.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 2)
ccslow = ccall.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 10)

cc = (ccfast - ccslow)/ccfast.std(0)
idx = cc.loc[0:3].mean(0).sort_values().index.values
cc = cc[idx]

pairs = idx[-int(len(idx)*0.01):]
pairs2 = idx[:int(len(idx)*0.01)]




###########################################################################################
# HD NEURONS
###########################################################################################
hd_neurons = mapping[mapping['hd']==1].index.values
hd_pairs = [p for p in pairs if p[0] in hd_neurons and p[1] in hd_neurons]
hd_pairs2 = [p for p in pairs2 if p[0] in hd_neurons and p[1] in hd_neurons]
hdp_neurons = np.unique(np.array(hd_pairs).flatten())

H = mapping.loc[hd_neurons, 'peak'].values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)
RGB = pd.DataFrame(index = hd_neurons, data = RGB)

figure()
subplot(221)
scatter(mapping['y'], mapping['x'], color = 'grey', alpha = 0.5, ec = None)
scatter(mapping.loc[hd_neurons,'y'], mapping.loc[hd_neurons,'x'], c = RGB.values)
scatter(mapping.loc[hdp_neurons,'y'], mapping.loc[hdp_neurons,'x'], c = 'white', s = 1)
subplot(222)
plot(np.cos(mapping.loc[hd_neurons,'peak']), np.sin(mapping.loc[hd_neurons,'peak']), '.', color = 'grey')

prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
            shrinkA=0,shrinkB=0, color = 'grey', alpha = 0.4)
for p in hd_pairs:
	# plot(np.cos(mapping.loc[list(p),'peak']), np.sin(mapping.loc[list(p),'peak']), '-', alpha = 0.5, color = 'grey')
	xystart = np.array([np.cos(mapping.loc[p[1],'peak']),np.sin(mapping.loc[p[1],'peak'])])
	xyend = np.array([np.cos(mapping.loc[p[0],'peak']),np.sin(mapping.loc[p[0],'peak'])])
	dxy = xyend - xystart
	xyend = xystart + 0.9*dxy
	annotate('', xyend, xystart, arrowprops=prop)

subplot(224)
hd_pairs_wtrep = [p for p in hd_pairs if (p[1],p[0]) not in list(hd_pairs)]

plot(np.cos(mapping.loc[hd_neurons,'peak']), np.sin(mapping.loc[hd_neurons,'peak']), '.', color = 'grey')
prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
            shrinkA=0,shrinkB=0, color = 'grey', alpha = 0.4)
for p in hd_pairs_wtrep:
	# plot(np.cos(mapping.loc[list(p),'peak']), np.sin(mapping.loc[list(p),'peak']), '-', alpha = 0.5, color = 'grey')
	xystart = np.array([np.cos(mapping.loc[p[1],'peak']),np.sin(mapping.loc[p[1],'peak'])])
	xyend = np.array([np.cos(mapping.loc[p[0],'peak']),np.sin(mapping.loc[p[0],'peak'])])
	dxy = xyend - xystart
	xyend = xystart + 0.9*dxy
	annotate('', xyend, xystart, arrowprops=prop)

subplot(223, projection = 'polar')
alpha = np.array([mapping.loc[list(p),'peak'].diff().values[-1] for p in hd_pairs])
alpha[alpha>np.pi] = 2*np.pi - alpha[alpha>np.pi]
alpha[alpha<-np.pi] = 2*np.pi + alpha[alpha<-np.pi]
bins = np.linspace(-np.pi, np.pi, 24)
x,_ = np.histogram(alpha, bins)
tmp = mapping.loc[hd_neurons, 'peak'].values
tmp = np.vstack(tmp) - tmp
tmp = tmp[np.triu_indices_from(tmp)]
tmp[tmp>np.pi] -= 2*np.pi
tmp[tmp<-np.pi] += 2*np.pi
yc,_ = np.histogram(tmp, bins)
# hist(alpha, bins)
plot(bins[0:-1], x/(yc+1))


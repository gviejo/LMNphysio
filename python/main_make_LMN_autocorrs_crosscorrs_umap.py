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
from itertools import combinations

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
# datasets1 = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets2 = np.loadtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.hstack((datasets1, datasets2))
datasets = np.loadtxt(os.path.join(data_directory,'datasets_KS25.txt'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)



allauto = {e:[] for e in ['wak', 'rem', 'sws']}
allfrates = {e:[] for e in ['wak', 'rem', 'sws']}
hd_index = []
shanks_index = []
allcc_wak = []
allcc_rem = []
allcc_sws = []
allpairs = []
alltcurves = []
allpeaks = []


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
	
		
	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################
	tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
	tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

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
	# tokeep = mean_frate[(mean_frate.loc[tokeep]>4).all(1)].index.values
	tokeep = mean_frate[mean_frate.loc[tokeep,'sws']>2].index.values


	spikes = {n:spikes[n] for n in tokeep}
	neurons = [name+'_'+str(n) for n in spikes.keys()]
	hd_index.append([n for n in neurons if int(n.split('_')[1]) in tokeep])

	############################################################################################### 
	# COMPUTE AUTOCORRS
	###############################################################################################
	for e, ep, tb, tw in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep], [1, 1, 1], [2000, 2000, 2000]):
		autocorr, frates = compute_AutoCorrs(spikes, ep, tb, tw)
		autocorr.columns = pd.Index(neurons)
		frates.index = pd.Index(neurons)
		allauto[e].append(autocorr)
		allfrates[e].append(frates)

	############################################################################################### 
	# CROSS CORRELATION
	###############################################################################################
	cc_wak = compute_CrossCorrs(spikes, wake_ep, norm=True)
	cc_rem = compute_CrossCorrs(spikes, rem_ep, norm=True)	
	cc_sws = compute_CrossCorrs(spikes, sws_ep, 2, 2000, norm=True)

	

	cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_rem = cc_rem.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)

	tcurves 							= tuning_curves[tokeep]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
	tcurves 							= tcurves[peaks.index.values]
	neurons 							= [name+'_'+str(n) for n in tcurves.columns.values]
	peaks.index							= pd.Index(neurons)
	tcurves.columns						= pd.Index(neurons)

	new_index = [(name+'_'+str(i),name+'_'+str(j)) for i,j in cc_wak.columns]
	cc_wak.columns = pd.Index(new_index)
	cc_rem.columns = pd.Index(new_index)
	cc_sws.columns = pd.Index(new_index)
	pairs = pd.Series(index = new_index, data = np.nan)
	for i,j in pairs.index:	
		if i in neurons and j in neurons:
			a = peaks[i] - peaks[j]
			pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))


	pairs = pairs.dropna().sort_values()




	#######################
	# SAVING
	#######################
	alltcurves.append(tcurves)
	allpairs.append(pairs)
	allcc_wak.append(cc_wak[pairs.index])
	allcc_rem.append(cc_rem[pairs.index])
	allcc_sws.append(cc_sws[pairs.index])
	allpeaks.append(peaks)


####################################
# CONCATENATING
####################################
for e in allauto.keys():
	allauto[e] = pd.concat(allauto[e], 1)
	allfrates[e] = pd.concat(allfrates[e])

frates = pd.DataFrame.from_dict(allfrates)
hd_index = np.hstack(hd_index)

alltcurves 	= pd.concat(alltcurves, 1)
allpairs 	= pd.concat(allpairs, 0)
allcc_wak 	= pd.concat(allcc_wak, 1)
allcc_rem 	= pd.concat(allcc_rem, 1)
allcc_sws 	= pd.concat(allcc_sws, 1)
allpeaks 	= pd.concat(allpeaks, 0)

#####################################

allindex = []
halfauto = {}
for e in allauto.keys():
	# 1. starting at 2
	auto = allauto[e].loc[0:]
	# auto = allauto[e]
	# 3. lower than 100 
	auto = auto.drop(auto.columns[auto.apply(lambda col: col.max() > 20.0)], axis = 1)
	# # 4. gauss filt
	auto = auto.rolling(window = 10, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 2.0)
	auto = auto.loc[0:40]
	# Drop nan
	auto = auto.dropna(1)
	halfauto[e] = auto
	allindex.append(auto.columns)


neurons = np.intersect1d(np.intersect1d(allindex[0], allindex[1]), allindex[2])
# neurons = np.intersect1d(neurons, fr_index)

# shanks_index = pd.concat(shanks_index)

# data = np.hstack([halfauto[e][neurons].values.T for e in halfauto.keys()])
data = np.hstack([halfauto[e][neurons].values.T for e in ['wak', 'sws']])
# data = halfauto['wak'].values.T
# neurons = halfauto['wak'].columns.values


from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


U = UMAP(n_neighbors = 15, n_components = 2).fit_transform(data)
K = KMeans(n_clusters = 2, random_state = 0).fit(U).labels_


figure()
scatter(U[:,0], U[:,1], c = K)

cc_grps = {}
for i in np.unique(K):
	index = neurons[K==i]
	sess_groups = pd.DataFrame(pd.Series({k:k.split("_")[0] for k in index})).groupby(0).groups	
	pairs = []
	for s in sess_groups.keys():
		tmp = np.sort([int(n.split('_')[-1]) for n in sess_groups[s]])
		tmp2 = [s+'_'+str(n) for n in tmp]
		[pairs.append(p) for p in combinations(tmp2, 2)]
	
	cc_grps[i] = {}
	for ep, cc in zip(['wak', 'rem', 'sws'], [allcc_wak, allcc_rem, allcc_sws]):
		cc_grps[i][ep] = cc[pairs]


from matplotlib.gridspec import GridSpec
titles = ['wake', 'REM', 'NREM']
figure()
gs = GridSpec(len(np.unique(K)),3)
for i in cc_grps.keys():
	pairs = allpairs.loc[cc_grps[i]['wak'].columns.values].sort_values().index.values
	for j, ep in enumerate(cc_grps[i].keys()):
		subplot(gs[i,j])		
		tmp = cc_grps[i][ep][pairs].T.values
		imshow(scipy.ndimage.gaussian_filter(tmp, 4), aspect = 'auto', cmap = 'jet')


figure()
count = 1
for j in range(len(np.unique(K))):
	for e in allauto.keys():	
		subplot(len(np.unique(K)),3,count)
		tmp = allauto[e][neurons[K==j]]
		plot(tmp.mean(1).loc[-100:100])
		count += 1
		title(e)
show()

figure()
count = 1
for e in allauto.keys():	
	subplot(1,3,count)
	for j in range(len(np.unique(K))):		
		tmp = allauto[e][neurons[K==j]]
		plot(tmp.mean(1).loc[-100:100])
	count += 1
	title(e)
show()

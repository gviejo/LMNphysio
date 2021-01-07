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
from sklearn.cluster import KMeans
from umap import UMAP
from matplotlib import gridspec


data_directory = '/mnt/DataGuillaume/'

############################################################################################### 
# DATASETS
###############################################################################################
alldata = {
	'ADN': ['LMN-ADN/A5001/'+s for s in ['A5001-200210B', 'A5001-200210C']],
	'DTN': ['DTN/A4002/'+s for s in ['A4002-200121A']],
	'LMN': ['LMN-ADN/A5002/'+s for s in ['A5002-200302A', 'A5002-200303A', 'A5002-200303B']]
		+['LMN/A1407/'+s for s in ['A1407-190416', 'A1407-190417', 'A1407-190422']],
	'POS': ['LMN-POSTSUB/A3004/'+s for s in ['A3004-200115A', 'A3004-200116D',
		'A3004-200117C', 'A3004-200117D', 'A3004-200118A', 'A3004-200118B']]
}

infos = getAllInfos(data_directory, np.hstack([alldata[p] for p in alldata.keys()]))

mapping = []
ccall = []
alltcurves = []
pairs = {k:[] for k in alldata.keys()}

which_shank = {'ADN':6, 'LMN':3, 'POS':0}

for p in alldata.keys():
	for s in alldata[p]:
		print(s)
		name 			= s.split('/')[-1]
		path 			= os.path.join(data_directory, s)
		episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
		events 			= list(np.where(episodes == 'wake')[0].astype('str'))
		spikes, shank 	= loadSpikeData(path)
		n_channels, fs, shank_to_channel 	= loadXML(path)
		position		= loadPosition(path, events, episodes)
		wake_ep 		= loadEpoch(path, 'wake', episodes)
		
		# TO RESTRICT BY SHANK		
		if p in which_shank:
			spikes 			= {n:spikes[n] for n in np.where(shank==which_shank[p])[0]}	
		neurons 		= [name+'_'+str(n) for n in spikes.keys()]

		######################
		# TUNING CURVES
		######################
		tcurves 		= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)	
		tcurves 		= smoothAngularTuningCurves(tcurves, 20, 2)
		tokeep, stat 	= findHDCells(tcurves, z = 10)		
		peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
		tcurves.columns						= pd.Index(neurons)
		alltcurves.append(tcurves)
			
		######################
		# SYNAPTIC CONNECTION
		######################
		mapp = pd.DataFrame(index = neurons,
			columns = ['hd', 'peak', 'structure'], 
			data = alldata[p].index(s))

		spks = spikes
		binsize = 0.5
		nbins = 100
		times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

		from itertools import product

		cc = pd.DataFrame(index = times, columns = list(product(neurons, neurons)))
		# cc2 = pd.DataFrame(index = times, columns = list(product(spikes.keys(), spikes.keys())))
			
		for i,j in cc.columns:
			if i != j:
				spk1 = spks[int(i.split("_")[1])].as_units('ms').index.values
				spk2 = spks[int(j.split("_")[1])].as_units('ms').index.values		
				tmp = crossCorr(spk1, spk2, binsize, nbins)					
				cc[(i,j)] = tmp			
		cc = cc.dropna(1)
		
		mapp['hd'] = 0
		mapp.loc[[name+'_'+str(n) for n in tokeep],'hd'] = 1	
		mapp.loc[[name+'_'+str(n) for n in peaks.index],'peak'] = peaks.values
		mapp['structure'] = p
		mapping.append(mapp)
		ccall.append(cc)
		for i in cc.columns.values:
			pairs[p].append(i)


mapping = pd.concat(mapping)
ccall = pd.concat(ccall,1)
alltcurves 	= pd.concat(alltcurves, 1)



gs = gridspec.GridSpec(3,4)
tmp = {}
alphas = []
for i,p in enumerate(['DTN', 'LMN', 'ADN', 'POS']):
	# SYNAPTIC DETECTION
	ccfast = ccall[pairs[p]].rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 1)
	ccslow = ccall[pairs[p]].rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 10)
	cc = (ccfast - ccslow)/ccfast.std(0)
	idx = cc.loc[0.5:3].mean(0).sort_values().index.values
	cc = cc[idx]
	pairs2 = idx[-int(len(idx)*0.3):]
	# pairs3 = idx[:int(len(idx)*0.1)]

	subplot(gs[0,i])
	plot(cc[pairs2])
	title(p)

	subplot(gs[1,i])
	imshow(cc.T, aspect = 'auto')
	axhline(cc.shape[1]-len(pairs2))

	subplot(gs[2,i], projection='polar')
	alpha = np.array([mapping.loc[list(p),'peak'].diff().values[-1] for p in pairs2])
	alpha[alpha>np.pi] = 2*np.pi - alpha[alpha>np.pi]
	alpha[alpha<-np.pi] = 2*np.pi + alpha[alpha<-np.pi]
	hist(alpha, 20)

	if p != 'DTN':
		alphas.append(alpha)

	tmp[p] = cc[pairs2].mean(1)

tmp = pd.DataFrame.from_dict(tmp)

figure()
subplot(121)
for i in tmp:
	plot(tmp[i], label = i)
legend()
subplot(122, projection = 'polar')
hist(np.hstack(alphas), 30)


figure()
gs = gridspec.GridSpec(2,8)

for i,p in enumerate(['DTN', 'LMN', 'ADN', 'POS']):
	# SYNAPTIC DETECTION
	ccfast = ccall[pairs[p]].rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 1)
	ccslow = ccall[pairs[p]].rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 10)
	cc = (ccfast - ccslow)/ccfast.std(0)
	idx = cc.loc[0.5:3].mean(0).sort_values().index.values
	cc = cc[idx]
	pairs2 = idx[-int(len(idx)*0.3):]
	# pairs3 = idx[:int(len(idx)*0.1)]

	subplot(gs[0,i])
	plot(cc[pairs2])
	title(p)

	subplot(gs[1,i])
	imshow(cc.T, aspect = 'auto')
	axhline(cc.shape[1]-len(pairs2))

subplot(gs[:,4:])
for i in tmp:
	plot(tmp[i], label = i)
legend()

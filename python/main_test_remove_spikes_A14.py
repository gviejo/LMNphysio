import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from itertools import product

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)


tocut = np.arange(0,1,0.1)

allcc_sws = {}
allpairs = {}
alltcurves = {}
allfrates = {}
allpeaks = {}


for cut in tocut:
	allcc_sws[cut] = []
	allpairs[cut] = []
	allfrates[cut] = []

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
			spikes = {n:spikes[n] for n in np.intersect1d(np.where(shank.flatten()>2)[0], np.where(shank.flatten()<4)[0])}		

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
			# tcurves_half = computeLMNAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])[0][1]
			tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
			tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
			tokeep, stat = findHDCells(tcurves_half)
			tokeep2.append(tokeep)
			stats2.append(stat)
			tcurves2.append(tcurves_half)

		tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
		
		lmn = tokeep		

		
		############################################################################################### 
		# CROSS CORRELATION
		###############################################################################################		
		newspikes = {n:[] for n in lmn}

		for n in lmn:
			tmp = spikes[n].restrict(sws_ep).as_series().sample(frac=1-cut).sort_index()
			newspikes[n] = nts.Ts(tmp)

		# compute firing rate for sws_ep
		mean_frate 	= computeMeanFiringRate(newspikes, [sws_ep], ['sws'])
		lmn = mean_frate.loc[tokeep][mean_frate.loc[tokeep,'sws']>2].index.values
		newspikes = {n:newspikes[n] for n in lmn}

		cc_sws = compute_CrossCorrs(newspikes, sws_ep, 5, 2000, norm=True)
		cc_sws = cc_sws.rolling(window=100, win_type='gaussian', center = True, min_periods = 1).mean(std = 10.0)

		tcurves 							= tuning_curves[lmn]
		peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
		tcurves 							= tcurves[peaks.index.values]
		neurons 							= [name+'_'+str(n) for n in tcurves.columns.values]
		peaks.index							= pd.Index(neurons)
		tcurves.columns						= pd.Index(neurons)

		new_index = [(name+'_'+str(i),name+'_'+str(j)) for i,j in cc_sws.columns]
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
		allpairs[cut].append(pairs)
		allcc_sws[cut].append(cc_sws[pairs.index])
		# allpeaks.append(peaks)
	
	allpairs[cut] 	= pd.concat(allpairs[cut], 0)
	allcc_sws[cut] 	= pd.concat(allcc_sws[cut], 1)
	
	idx = allpairs[cut].sort_values().index.values
	allcc_sws[cut] 	= allcc_sws[cut][idx]


import matplotlib.gridspec as gsp

figure()
for i, c in enumerate(allpairs.keys()):	
	subplot(2,5,i+1)
	cc = allcc_sws[c]
	imshow(scipy.ndimage.gaussian_filter(cc.values.T, 4), aspect = 'auto', cmap = 'jet')
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])
	title(c)


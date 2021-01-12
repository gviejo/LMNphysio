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


tocut = [0,1,2,3,4]
binsizes = [10, 20, 30, 40, 50, 60]
params = list(product(tocut, binsizes))

allcc_sws = {}
allpairs = {}
alltcurves = {}
allfrates = {}
allpeaks = {}


for cut in tocut:
	allcc_sws[cut] = {}
	allpairs[cut] = {}
	allfrates[cut] = {}
	for bin_size in binsizes:
		allcc_sws[cut][bin_size] = []
		allpairs[cut][bin_size] = []
		allfrates[cut][bin_size] = []

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
			bins = np.arange(sws_ep['start'].iloc[0], sws_ep['end'].iloc[-1] + bin_size*1000, bin_size*1000)
			rate = []
			idx_spk = {}
			for i,n in enumerate(lmn):
				count, _ = np.histogram(spikes[n].index.values, bins)
				rate.append(count)
				idx_spk[n] = nts.Tsd(t = spikes[n].index.values, d = np.digitize(spikes[n].index.values, bins))
				
			rate = np.array(rate).T

			newspikes = {n:[] for n in lmn}

			# Pure WTA version
			# idx = np.where((rate==np.vstack(rate.max(1))).sum(1) == 1)[0] + 1 #wta
			# idx_n = lmn[np.argmax(rate[idx],1)]
			# for n in lmn:
			# 	t = idx_spk[n][idx_spk[n].as_series().isin(idx_bin)].index.values
			# 	newspikes[n] = nts.Ts(t = t)
			for i,n in enumerate(lmn):
				idx = np.where(rate[:,i] > cut)[0]+1
				t = idx_spk[n][idx_spk[n].as_series().isin(idx)].index.values
				newspikes[n] = nts.Ts(t = t)

			# compute firing rate for sws_ep
			mean_frate 	= computeMeanFiringRate(newspikes, [sws_ep], ['sws'])
			lmn = mean_frate.loc[tokeep][mean_frate.loc[tokeep,'sws']>1].index.values
			newspikes = {n:newspikes[n] for n in lmn}

			cc_sws = compute_CrossCorrs(newspikes, sws_ep, 2, 2000, norm=True)
			cc_sws = cc_sws.rolling(window=50, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)

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
			allpairs[cut][bin_size].append(pairs)
			allcc_sws[cut][bin_size].append(cc_sws[pairs.index])
			# allpeaks.append(peaks)
		
		allpairs[cut][bin_size] 	= pd.concat(allpairs[cut][bin_size], 0)
		allcc_sws[cut][bin_size] 	= pd.concat(allcc_sws[cut][bin_size], 1)
		
		idx = allpairs[cut][bin_size].sort_values().index.values
		allcc_sws[cut][bin_size] 	= allcc_sws[cut][bin_size][idx]


import matplotlib.gridspec as gsp

figure()
gs = gsp.GridSpec(len(tocut),len(binsizes))
for i, c in enumerate(allpairs.keys()):	
	for j, b in enumerate(allpairs[c].keys()):
		subplot(gs[i,j])
		cc = allcc_sws[c][b]
		imshow(scipy.ndimage.gaussian_filter(cc.values.T, 2), aspect = 'auto', cmap = 'jet')
		xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])
		title(str(c) + ' | ' + str(b) + 'ms')


import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)


allcc_wak = []
allcc_rem = []
allcc_sws = []
allpairs = []
alltcurves = []
allfrates = []
allvcurves = []
allscurves = []
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

	# NEURONS FROM ADN	
	if 'A5011' in s:
		adn = np.where(shank <=3)[0]
		lmn = np.where(shank >3)[0]

	adn = np.intersect1d(adn, tokeep)
	lmn = np.intersect1d(lmn, tokeep)


	# spikes = {n:spikes[n] for n in tokeep}

	# 0 is lmn
	# 1 is adn
	tmp = {}
	tmp[0] = nts.Ts(np.sort(np.hstack([spikes[n].index.values for n in lmn])))
	tmp[1] = nts.Ts(np.sort(np.hstack([spikes[n].index.values for n in adn])))
	
	spikes = tmp
	
	############################################################################################### 
	# CROSS CORRELATION
	###############################################################################################
	cc_wak = compute_CrossCorrs(spikes, wake_ep, norm=True)
	cc_rem = compute_CrossCorrs(spikes, rem_ep, norm=True)	
	cc_sws = compute_CrossCorrs(spikes, sws_ep, 5, 10000, norm=True)


	cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_rem = cc_rem.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)

	
	#######################
	# SAVING
	#######################
	allcc_wak.append(cc_wak)
	allcc_rem.append(cc_rem)
	allcc_sws.append(cc_sws)


allcc_wak 	= pd.concat(allcc_wak, 1)
allcc_rem 	= pd.concat(allcc_rem, 1)
allcc_sws 	= pd.concat(allcc_sws, 1)



figure()
for i in range(2):
	subplot(2,2,i+1)
	plot(allcc_sws.iloc[:,i])
for i in range(2):
	subplot(2,2,i+1+2)
	plot(allcc_wak.iloc[:,i])

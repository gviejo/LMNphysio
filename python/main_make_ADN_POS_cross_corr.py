import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from itertools import product

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_ADN_POS.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

ccall 			= []
tcurves 		= []
peaks 			= []
mapping 		= []


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
	sleep_ep		= loadEpoch(path, 'sleep')

	neurons 		= [name+'_'+str(n) for n in spikes.keys()]
	mapp = pd.DataFrame(index = neurons,
		columns = ['session', 'shank', 'channel', 'hd'], 
		data = list(datasets).index(s))
	mapp['shank'] = shank
	mapp['hd'] = 0
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
			
	mapp.loc[[name+'_'+str(n) for n in tokeep],'hd'] = 1

	######################
	# NORMAL CROSS-CORR
	######################
	spks = spikes
	binsize = 0.5
	nbins = 100
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

	cc = pd.DataFrame(index = times, columns = list(product(neurons, neurons)))	
	
	for i,j in cc.columns:
		if i != j:
			spk1 = spks[int(i.split("_")[1])].restrict(wake_ep).as_units('ms').index.values
			spk2 = spks[int(j.split("_")[1])].restrict(wake_ep).as_units('ms').index.values		
			tmp = crossCorr(spk1, spk2, binsize, nbins)					
			cc[(i,j)] = tmp			
	cc = cc.dropna(1)	

	######################
	# TOSAVE
	######################
	tcurves.append(tcurve)		
	ccall.append(cc)
	peaks.append(peak)
	mapping.append(mapp)


tcurves 		= pd.concat(tcurves, 1)
ccall 			= pd.concat(ccall, 1)
peaks 			= pd.concat(peaks)
mapping 		= pd.concat(mapping)

ccfast = ccall.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 1)
ccslow = ccall.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 10)
cc = (ccfast - ccslow)/ccfast.std(0)



hd_adn = mapping[(mapping['shank']>=8)&(mapping['hd']==1)].index.values
hd_pos = mapping[(mapping['shank']<8)&(mapping['shank']>3)&(mapping['hd']==1)].index.values
nhd_pos = mapping[(mapping['shank']<8)&(mapping['shank']>3)&(mapping['hd']==0)].index.values

pairs = {k:[] for k in ['hd_adn', 'hd_pos', 'nhd_pos', 'hd_pos_adn', 'nhd_pos_adn']}
for p in cc.columns.values:
	if p[0] in hd_adn and p[1] in hd_adn:
		pairs['hd_adn'].append(p)
	if p[0] in hd_pos and p[1] in hd_pos:
		pairs['hd_pos'].append(p)
	if p[0] in nhd_pos and p[1] in nhd_pos:
		pairs['nhd_pos'].append(p)
	if (p[0] in hd_adn and p[1] in hd_pos) or ((p[0] in hd_pos and p[1] in hd_adn)):		
		pairs['hd_pos_adn'].append(p)
	if (p[0] in hd_adn and p[1] in nhd_pos) or ((p[0] in nhd_pos and p[1] in hd_adn)):
		pairs['nhd_pos_adn'].append(p)


figure()
subplot(231)
plot(cc[pairs['hd_adn']])
subplot(232)
plot(cc[pairs['hd_pos']])
subplot(233)
plot(cc[pairs['nhd_pos']])
subplot(234)
plot(cc[pairs['hd_pos_adn']])
subplot(232)
plot(cc[pairs['nhd_pos_adn']])



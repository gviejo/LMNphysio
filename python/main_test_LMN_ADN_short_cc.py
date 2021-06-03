import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import itertools

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

s = 'LMN-ADN/A5011/A5011-201014A'



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
	lmn = np.where(shank ==5)[0]

adn 	= np.intersect1d(adn, tokeep)
lmn 	= np.intersect1d(lmn, tokeep)
tokeep 	= np.hstack((adn, lmn))
spikes 	= {n:spikes[n] for n in tokeep}

tcurves 		= tuning_curves[tokeep]
peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))


#############################################################################################
# FIGURES
#############################################################################################
adn = peaks[adn].sort_values().index.values
lmn = peaks[lmn].sort_values().index.values

figure()
ax = subplot(211)
for k, n in enumerate(adn):
	plot(spikes[n].fillna(peaks[n]), '|')	
tmp = position['ry']
tmp	= tmp.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)	
plot(tmp, linewidth = 2, color = 'black')
ylim(0, 2*np.pi)

subplot(212, sharex = ax)
for k, n in enumerate(lmn):
	plot(spikes[n].fillna(peaks[n]), '|')	
tmp = position['ry']
tmp	= tmp.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)	
plot(tmp, linewidth = 2, color = 'black')
ylim(0, 2*np.pi)



	
############################################################################################### 
# CROSS CORRELATION
###############################################################################################
pairs = list(itertools.product(lmn, adn))
cc_wak = []
cc_sws = []
for p in pairs:
	cc_wak.append(compute_PairCrossCorr(spikes, wake_ep, p, 0.5, 500, True))
	cc_sws.append(compute_PairCrossCorr(spikes, sws_ep, p, 0.5, 500, True))

cc_wak = pd.concat(cc_wak, 1)
cc_sws = pd.concat(cc_sws, 1)
cc_wak.columns = pairs
cc_sws.columns = pairs

############################################################################################### 
# CROSS-CORRELATION / AHV
###############################################################################################




for i, m in enumerate(lmn):
	print(i)
	figure()
	gs = GridSpec(3, len(adn))
	for j, n in enumerate(adn):
		subplot(gs[0,j], projection='polar')
		plot(tuning_curves[m], label = 'lmn')
		plot(tuning_curves[n], label = 'adn')
		legend()
		subplot(gs[1,j])
		plot(cc_wak[(m,n)].loc[-30:30])
		title('wake')
		subplot(gs[2,j])
		plot(cc_sws[(m,n)].loc[-30:30])
		title('sws')
	show(block=True)	

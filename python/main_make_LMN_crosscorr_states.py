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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
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
# for s in ['A5000/A5002/A5002-200304A']:	
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
	tuning_curves = {1:computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)}
	for i in tuning_curves:
		tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)

	# CHECKING HALF EPOCHS
	wake2_ep = splitWake(wake_ep)
	tokeep2 = []
	stats2 = []
	tcurves2 = []
	for i in range(2):
		# tcurves_half = computeLMNAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])[0][1]
		tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
		tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)
		tokeep, stat = findHDCells(tcurves_half)
		tokeep2.append(tokeep)
		stats2.append(stat)
		tcurves2.append(tcurves_half)

	tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
	tokeep2 = np.union1d(tokeep2[0], tokeep2[1])

	# Checking firing rate
	spikes = {n:spikes[n] for n in tokeep}
	mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])	
	# tokeep = mean_frate[(mean_frate.loc[tokeep]>4).all(1)].index.values
	tokeep = mean_frate[mean_frate.loc[tokeep,'sws']>4].index.values

	############################################################################################### 
	# CROSS CORRELATION
	###############################################################################################
	cc_wak = compute_CrossCorrs(spikes, wake_ep, norm=True)
	cc_rem = compute_CrossCorrs(spikes, rem_ep, norm=True)	
	cc_sws = compute_CrossCorrs(spikes, sws_ep, 2, 2000, norm=True)

	

	cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_rem = cc_rem.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)

	tcurves 							= tuning_curves[1][tokeep]
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

 
alltcurves 	= pd.concat(alltcurves, 1)
allpairs 	= pd.concat(allpairs, 0)
allcc_wak 	= pd.concat(allcc_wak, 1)
allcc_rem 	= pd.concat(allcc_rem, 1)
allcc_sws 	= pd.concat(allcc_sws, 1)
allpeaks 	= pd.concat(allpeaks, 0)


allpairs = allpairs.sort_values()


sess_groups = pd.DataFrame(pd.Series({k:k.split("_")[0] for k in alltcurves.columns.values})).groupby(0).groups


colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(sess_groups)))

datatosave = {	'tcurves':alltcurves,
				'sess_groups':sess_groups,
				'cc_wak':allcc_wak,
				'cc_rem':allcc_rem,
				'cc_sws':allcc_sws,
				'pairs':allpairs,
				'peaks':allpeaks
				}

# cPickle.dump(datatosave, open(os.path.join('../figures/figures_ipn_2020', 'All_crosscor.pickle'), 'wb'))


##########################################################
# TUNING CURVES
figure()
count = 1
for i, g in enumerate(sess_groups.keys()):
	for j, n in enumerate(sess_groups[g]):
		subplot(13,20,count, projection = 'polar')
		plot(alltcurves[n], color = colors[i])
		# title(n)
		xticks([])
		yticks([])
		count += 1

##########################################################
# CROSS CORR
titles = ['wake', 'REM', 'NREM']
figure()
subplot(2,4,1)
plot(allpairs.values, np.arange(len(allpairs))[::-1])
for i, cc in enumerate([allcc_wak, allcc_rem, allcc_sws]):
	subplot(2,4,i+2)
	tmp = cc[allpairs.index].T.values
	imshow(scipy.ndimage.gaussian_filter(tmp, 2), aspect = 'auto', cmap = 'jet')

	title(titles[i])
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])

	subplot(2,3,i+3+1)
	plot(cc, alpha = 0.5)


##########################################################
# EXEMPLES
groups = allpairs.groupby(np.digitize(allpairs, [0, np.pi/3, 2*np.pi/3, np.pi])).groups

figure()
for i, g in enumerate(groups.keys()):
	for j, cc in enumerate([allcc_wak, allcc_rem, allcc_sws]):
		subplot(3,3,j+1+i*3)
		plot(cc[groups[g]], color = 'grey', alpha = 0.6)
		if i == 0:
			title(titles[j])

show()





cc = allcc_sws[allpairs.index]
tmp1 = cc.loc[:0].rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 10.0)
tmp2 = cc.loc[0:].rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 10.0)
cc2 = pd.concat((tmp1, tmp2))

figure()
imshow(scipy.ndimage.gaussian_filter(cc2.values.T, 2), aspect = 'auto', cmap = 'jet')
xticks([0, np.where(cc2.index.values == 0)[0][0], len(cc2)], [cc2.index[0], 0, cc2.index[-1]])



cc2 = allcc_sws[allpairs.index]

tmp = (np.vstack((cc2.loc[:-100].values,cc2.loc[100:].values))).T


figure()
imshow(scipy.ndimage.gaussian_filter(tmp, 10), aspect = 'auto', cmap = 'jet')

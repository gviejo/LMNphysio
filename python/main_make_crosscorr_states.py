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

data_directory 		= '/mnt/DataGuillaume/LMN-ADN/A5011'

# info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

sessions = ['A5011-201010A', 'A5011-201011A']

allcc_wak = []
allcc_rem = []
allcc_sws = []
allpairs = []
alltcurves = []
allfrates = []
allvcurves = []
allscurves = []
allpeaks = []



for s in sessions:
	path = os.path.join(data_directory, s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################	
	# episodes 							= info.filter(like='Trial').loc[s].dropna().values
	# events								= list(np.where(episodes == 'wake')[0].astype('str'))
	episodes 							= ['sleep', 'wake', 'sleep']
	events 								= ['1']
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sleep_ep 							= loadEpoch(path, 'sleep')					
	sws_ep								= loadEpoch(path, 'sws')
	rem_ep								= loadEpoch(path, 'rem')

	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################
	tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
	# spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 21)
	# autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
	# autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)
	# velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30)
	# mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])
	# speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)


	for i in tuning_curves:
		tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)

	# velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
	# speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

	# sorting by angular differences
	tokeep, stat 	= findHDCells(tuning_curves[1], z = 10, p = 0.001 , m = 1)

	# if s == 'A1407-190416':
	# 	tokeep = np.delete(tokeep, np.where(tokeep==5))
	# 	tokeep = np.delete(tokeep, np.where(tokeep==2))

	adn = list(np.where(shank <= 3)[0])
	lmn = list(np.where(shank == 5)[0])
	adn = np.intersect1d(adn, tokeep)
	lmn = np.intersect1d(lmn, tokeep)

	# spikes = {n:spikes[n] for n in np.hstack([adn, lmn])}
	spikes = {n:spikes[n] for n in lmn}

	# for i, n in enumerate(adn):
	# 	subplot(3,4,i+1, projection = 'polar')
	# 	plot(tuning_curves[1][n])	
			
	# for i, n in enumerate(lmn):
	# 	subplot(3,4,i+1, projection = 'polar')
	# 	plot(tuning_curves[1][n])	


	
	############################################################################################### 
	# CROSS CORRELATION
	###############################################################################################
	cc_wak = compute_CrossCorrs(spikes, wake_ep, norm=True)
	cc_rem = compute_CrossCorrs(spikes, rem_ep, norm=True)	
	cc_sws = compute_CrossCorrs(spikes, sws_ep, 2, 2000, norm=True)

	cc_wak = cc_wak.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 1.0)
	cc_rem = cc_rem.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 1.0)
	cc_sws = cc_sws.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 1.0)



	tcurves 							= tuning_curves[1][tokeep]
	# velo_curves 						= velo_curves[tokeep]
	# speed_curves						= speed_curves[tokeep]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
	tcurves 							= tcurves[peaks.index.values]
	# velo_curves							= velo_curves[peaks.index.values]
	# speed_curves 						= speed_curves[peaks.index.values]
	# mean_frate							= mean_frate.loc[peaks.index.values]
	neurons 							= [s+'_'+str(n) for n in tcurves.columns.values]
	peaks.index							= pd.Index(neurons)
	tcurves.columns						= pd.Index(neurons)
	# velo_curves.columns 				= pd.Index(neurons)
	# speed_curves.columns 				= pd.Index(neurons)
	# mean_frate.index 					= pd.Index(neurons)
	

	new_index = [(s+'_'+str(i),s+'_'+str(j)) for i,j in cc_wak.columns]
	cc_wak.columns = pd.Index(new_index)
	cc_rem.columns = pd.Index(new_index)
	cc_sws.columns = pd.Index(new_index)
	pairs = pd.Series(index = new_index)
	for i,j in pairs.index:	
		if i in neurons and j in neurons:
			a = peaks[i] - peaks[j]
			pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))

	pairs = pairs.dropna().sort_values()

	# sys.exit()

	#######################
	# ONLY AD AND LMN
	#######################
	# tmp = list(product(adn, lmn))
	# tmp2 = [(s+'_'+str(i),s+'_'+str(j)) for i,j in tmp]

	# pairs = pairs[tmp2]


	# index = np.digitize(pairs.values, np.linspace(0, np.pi, 5))
	# for i in np.unique(index):
	# 	# subplot(2,2,i+1)
	# 	tmp = cc_sws[pairs[index==i].index]
	# 	tmp = tmp - tmp.mean(0)
	# 	tmp = tmp / tmp.std(0)
	# 	plot(tmp.mean(1), label = i)
	# legend()


	# sys.exit()
	#######################
	# SAVING
	#######################
	alltcurves.append(tcurves)
	# allvcurves.append(velo_curves)
	# allscurves.append(speed_curves)
	allpairs.append(pairs)
	allcc_wak.append(cc_wak[pairs.index])
	allcc_rem.append(cc_rem[pairs.index])
	allcc_sws.append(cc_sws[pairs.index])
	# allfrates.append(mean_frate)
	allpeaks.append(peaks)

 
alltcurves 	= pd.concat(alltcurves, 1)
# allscurves 	= pd.concat(allscurves, 1)
# allvcurves 	= pd.concat(allvcurves, 1)
allpairs 	= pd.concat(allpairs, 0)
# allfrates 	= pd.concat(allfrates, 0)
allcc_wak 	= pd.concat(allcc_wak, 1)
allcc_rem 	= pd.concat(allcc_rem, 1)
allcc_sws 	= pd.concat(allcc_sws, 1)
allpeaks 	= pd.concat(allpeaks, 0)



allpairs = allpairs.sort_values()
# allfrates = allfrates.astype('float')


index = np.digitize(allpairs.values, np.linspace(0, np.pi, 6))
for i in np.unique(index):
	# subplot(2,2,i+1)
	tmp = allcc_rem[allpairs[index==i].index]
	tmp = tmp - tmp.mean(0)
	tmp = tmp / tmp.std(0)
	plot(tmp.mean(1), label = i)
legend()





sess_groups = pd.DataFrame(pd.Series({k:k.split("_")[0] for k in alltcurves.columns.values})).groupby(0).groups

colors = ['blue', 'red', 'green']


datatosave = {	'tcurves':alltcurves,
				'sess_groups':sess_groups,
				'frates':allfrates,
				'cc_wak':allcc_wak,
				'cc_rem':allcc_rem,
				'cc_sws':allcc_sws,
				'pairs':allpairs,
				'peaks':allpeaks
				}

cPickle.dump(datatosave, open('../data/test_lmn_adn_crosscorr.pickle', 'wb'))


sys.exit()
##########################################################
# TUNING CURVES
figure()
for i, g in enumerate(sess_groups.keys()):
	for j, n in enumerate(sess_groups[g]):
		subplot(3,8,j+1+i*8, projection = 'polar')
		plot(alltcurves[n], color = colors[i])
		title(n)

##########################################################
# ANGULAR VELOCITY CURVEs
figure()
for i, g in enumerate(sess_groups.keys()):
	for j, n in enumerate(sess_groups[g]):
		subplot(3,8,j+1+i*8)
		plot(allvcurves[n], color = colors[i])


##########################################################
# SPEED CURVEs
figure()
for i, g in enumerate(sess_groups.keys()):
	for j, n in enumerate(sess_groups[g]):
		subplot(3,8,j+1+i*8)
		plot(allscurves[n], color = colors[i])

##########################################################
# CROSS CORR
titles = ['wake', 'REM', 'NREM']
figure()
# subplot(221)
# scatter(np.log(allfrates['wake'].values), np.log(allfrates['rem'].values))
# xlabel("Wake")
# ylabel("REM")
# subplot(222)
# scatter(np.log(allfrates['wake'].values), np.log(allfrates['sws'].values))
# xlabel("Wake")
# ylabel("SWS")
for i, cc in enumerate([allcc_wak, allcc_rem, allcc_sws]):
	subplot(2,3,i+1+3)
	imshow(cc[allpairs.index].T, aspect = 'auto', cmap = 'jet')
	title(titles[i])
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])



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








sys.exit()



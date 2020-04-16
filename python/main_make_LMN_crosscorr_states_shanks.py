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
data_directory = r'D:\Dropbox (Peyrache Lab)\Peyrache Lab Team Folder\Data\LMN'
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



# for s in datasets[0:-2]:
for s in ['A5000/A5002/A5002-200304A']:	
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
	tuning_curves = {1:computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)}
	for i in tuning_curves:
		tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)

	spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep.loc[[0]], 30)
	velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=False)
	speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)



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
	tokeep2 = np.union1d(tokeep2[0], tokeep2[1])
			
	############################################################################################### 
	# CROSS CORRELATION
	###############################################################################################

	spikes = {n:spikes[n] for n in np.where(shank==5)[0]}

	cc_wak = compute_CrossCorrs(spikes, wake_ep, 5, 200, norm=True)
	cc_rem = compute_CrossCorrs(spikes, rem_ep, 5, 200, norm=True)
	cc_sws = compute_CrossCorrs(spikes, sws_ep, 1, 200, norm=True)

	auto_wak,_ = compute_AutoCorrs(spikes, wake_ep, 5, 200)
	auto_rem,_ = compute_AutoCorrs(spikes, rem_ep, 5, 200)
	auto_sws,_ = compute_AutoCorrs(spikes, sws_ep, 1, 200)


	figure()
	subplot(2,3,1)
	plot(auto_wak)
	title('Wake')
	subplot(2,3,2)
	plot(auto_rem)
	title('REM')
	subplot(2,3,3)
	plot(auto_sws)
	title('SWS')

	subplot(2,3,4)
	plot(cc_wak)
	subplot(2,3,5)
	plot(cc_rem)
	subplot(2,3,6)
	plot(cc_sws)

	show()

	from matplotlib import gridspec
	figure()
	gs = gridspec.GridSpec(4,np.sum(shank==5))
	for i, n in enumerate(np.where(shank==5)[0]):
		subplot(gs[0,i], projection = 'polar')
		plot(tuning_curves[1][n])
		subplot(gs[1,i])
		plot(velo_curves[n])
		subplot(gs[2,i])
		plot(speed_curves[n])
		subplot(gs[3,i])
		imshow(spatial_curves[n], interpolation = 'bilinear')

	show()


	order = []


	sys.exit()


	cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_rem = cc_rem.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)


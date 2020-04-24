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

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = r'D:\Dropbox (Peyrache Lab)\Peyrache Lab Team Folder\Data\LMN'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)

infoall = []
ccufos = []

# for s in datasets:
# for s in datasets:
for s in ['A5000/A5002/A5002-200303B']:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))
	events			= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)
	sleep_ep		= loadEpoch(path, 'sleep')
	sws_ep 			= loadEpoch(path, 'sws')
	rem_ep 			= loadEpoch(path, 'rem')
	theta_wake_ep	= loadEpoch(path, 'wake.evt.theta')


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
		tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
		tokeep, stat = findHDCells(tcurves_half)
		tokeep2.append(tokeep)
		stats2.append(stat)
		tcurves2.append(tcurves_half)

	tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
	tokeep2 = np.union1d(tokeep2[0], tokeep2[1])

	tcurves 							= tuning_curves[1][tokeep]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
	tcurves 							= tcurves[peaks.index.values]
	neurons 							= [name+'_'+str(n) for n in spikes.keys()]

	info 								= pd.DataFrame(index = neurons, columns = ['shank', 'hd', 'peaks'], data = 0)
	info['shank'] 						= shank.flatten()
	info['peaks'].iloc[tokeep] 			= peaks.values
	info['hd'].iloc[tokeep]				= 1

	############################################################################################### 
	# SUM OF SPIKEs
	###############################################################################################
	phase = pd.read_hdf(os.path.join(path, 'Analysis', 'phase_theta_wake.h5'))
	phase = nts.Tsd(phase)
	phase = phase.restrict(wake_ep)

	peaks, throughs = getPeaksandTroughs(phase, 10)

	half1 = nts.IntervalSet(start = peaks.index.values[0:-1], end = peaks.index.values[0:-1]+np.diff(peaks.index.values)/3)
	half2 = nts.IntervalSet(start = peaks.index.values[0:-1]+np.diff(peaks.index.values)/3, end = peaks.index.values[1:]-np.diff(peaks.index.values)/3)
	half3 = nts.IntervalSet(start = peaks.index.values[1:]-np.diff(peaks.index.values)/3, end = peaks.index.values[1:])

	# tmp = peaks.index.values
	# dtmp = np.diff(peaks.index.values)
	# half1 = nts.IntervalSet(
	# 	start = tmp[0:-1]+dtmp/4,
	# 	end = tmp[1:]-dtmp/4)
	# half2 = nts.IntervalSet(
	# 	start = tmp[1:-1] - dtmp[0:-1]/4,
	# 	end = tmp[1:-1] + dtmp[1:]/4)

	sys.exit()




	autosh = {}	
	for i, sh in enumerate([2,3,4,5]):
	# for i, sh in enumerate([2]):
		print(sh)
		ss = pd.read_hdf(os.path.join(path, 'Analysis/SS_'+str(sh)+'.h5'))	
		ss = ss[ss.columns[0]]
		ss = ss.loc[wake_ep.start[0]:wake_ep.end[0]]

		sys.exit()

		# COMPUTING AUtocorr for half-cycles
		peaks, _ = scipy.signal.find_peaks(ss.values, threshold=4)
		peaks = nts.Tsd(ss.index.values[peaks])
		peaks = peaks.restrict(theta_wake_ep)

		auto1, fr = compute_AutoCorrs({0:peaks}, half1, 1, 100)
		auto2, fr = compute_AutoCorrs({0:peaks}, half2, 1, 100)
		auto3, fr = compute_AutoCorrs({0:peaks}, half3, 1, 100)

		# auto1 = auto1.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
		# auto2 = auto2.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
		# auto3 = auto3.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)

		subplot(2,2,i+1)
		plot(auto1)
		plot(auto2)
		plot(auto3)
		

	show()


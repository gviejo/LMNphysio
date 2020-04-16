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
	ufo_ep, ufo_tsd	= loadUFOs(path)

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
	autosh = {}	
	for sh in [2,3,4,5]:
		print(sh)
		ss = pd.read_hdf(os.path.join(path, 'Analysis/SS_'+str(sh)+'.h5'))
		ss = ss[ss.columns[0]]

		autocep = {}
		for n, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):
			auto2 = {}
			for t in range(1, 10):
				peaks = []
				for i in ep.index:
					tmp = ss.loc[ep.start[i]:ep.end[i]]
					peak, _ = scipy.signal.find_peaks(tmp.values, threshold=t)
					peaks.append(tmp.index.values[peak])

				peaks = nts.Ts(t=np.hstack(peaks))
				autoc, fr = compute_AutoCorrs({0:peaks}, ep, 1, 200)
				# autoc, fr = compute_AutoCorrs({0:peaks}, ep, 5, 200)
				auto2[t]= autoc[0]
			auto2 = pd.DataFrame.from_dict(auto2)
			# auto2 = auto2.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
			autocep[n] = auto2

		del ss
		autosh[sh] = autocep

	sys.exit()
	
	for sh in autosh.keys():
		figure()
		for i, e in enumerate(autosh[sh].keys()):
			subplot(2,3,i+1)
			tmp = autosh[sh][e].values.T
			imshow(tmp, aspect = 'auto')
			locator_params(nbins=6)
			subplot(2,3,i+1+3)
			plot(autosh[sh][e][5])
			title(e)
	show()

	
	from matplotlib import gridspec

	figure()	
	gs = gridspec.GridSpec(3,4)
	for i,sh in enumerate(autosh.keys()):
		for j,e in enumerate(autosh[sh].keys()):
			subplot(gs[j,i])
			plot(autosh[sh][e][5])
			ylabel(e)
			if j==0:				
				title('Shank '+str(sh))
	show()

	sys.exit()
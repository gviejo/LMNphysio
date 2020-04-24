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
import _pickle as cPickle

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
for s in ['A5000/A5002/A5002-200304A']:
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
	# THETA HISTOGRAM WAKE
	############################################################################################### 
	spikes_phase = cPickle.load(open(path+'/Analysis/spike_theta_wake.pickle', 'rb'))

	bins = np.linspace(0, 2*np.pi, 20)
	df = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = list(spikes_phase.keys()))

	for n in spikes_phase.keys():
		tmp = spikes_phase[n].values
		tmp += 2*np.pi
		tmp %= 2*np.pi
		a, b = np.histogram(tmp, bins, density=True)
		df[n] = a * np.diff(b)

	df = df.rolling(window = 40, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)

	# for i, s in enumerate(np.unique(shank)):
	# 	figure()
	# 	for j, n in enumerate(np.where(shank==s)[0]):
	# 		subplot(int(np.sqrt(np.sum(shank==s)))+1,int(np.sqrt(np.sum(shank==s)))+1,j+1)
	# 		plot(df[n])

	figure()
	for i, s in enumerate(np.unique(shank)):
		subplot(2,3,i+1)
		plot(df[np.where(shank==s)[0]])
	show()


	sys.exit()
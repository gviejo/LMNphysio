import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pingouin import partial_corr

def zscore_rate(rate):
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	return rate



data_directory = '/mnt/Data2/Opto/A8000/A8015/A8015-210826A'

episodes = ['sleep', 'wake', 'sleep', 'wake']
# events = ['1', '3']

# episodes = ['sleep', 'wake', 'sleep']
events = ['1', '3']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)


position 							= loadPosition(data_directory, events, episodes, 2, 1)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
sws_ep								= loadEpoch(data_directory, 'sws')
acceleration						= loadAuxiliary(data_directory)
# #sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

#################
# TUNING CURVES
tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60)
#tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tuning_curves 						= smoothAngularTuningCurves(tuning_curves, 10, 2)

#tokeep, stat 						= findHDCells(tuning_curves, z=50, p = 0.001)
tokeep = list(spikes.keys())
#tokeep = [0, 2, 4, 5]

tcurves 							= tuning_curves[tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]


#################
# OPTO
opto_ep 							= loadOptoEp(data_directory, epoch=0, n_channels = 2, channel = 0)
opto_ep 							= opto_ep.merge_close_intervals(40000)
frates, rasters, bins, stim_duration = computeRasterOpto(spikes, opto_ep, 1000)

opto_ep = sws_ep.intersect(opto_ep)

nopto_ep = nts.IntervalSet(
	start = opto_ep['start'] - (opto_ep['end'] - opto_ep['start']),
	end = opto_ep['start']
	)



###################
# ISI
opto_isi = compute_ISI(spikes, opto_ep, 2000, 30, log_=True)
nopto_isi = compute_ISI(spikes, nopto_ep, 2000, 30, log_=True)

figure()
for i, n in enumerate(tokeep):
	subplot(3,5,i+1)
	plot(opto_isi[n])
	plot(nopto_isi[n])

show()
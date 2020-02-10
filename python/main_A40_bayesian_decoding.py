import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean

# data_directory = '/mnt/DataGuillaume/LMN/A1410/A1410-200116A/A1410-200116A'
data_directory = '/mnt/LocalHDD/A4002/A4002-200121/A4002-200121_0'

# data_directory = '../data/A1400/A1407/A1407-190422'
# data_directory = '/mnt/DataGuillaume/PostSub/A3003/A3003-190516A'

# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['wake', 'sleep']
episodes = ['sleep', 'wake']
# events = ['1', '3']
events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
# acceleration						= loadAuxiliary(data_directory, 6)
# sleep_ep							= refineSleepFromAccel(acceleration, sleep_ep)


tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tcurves 							= smoothAngularTuningCurves(tuning_curves, 10, 2)
tokeep, stat 						= findHDCells(tcurves)
tcurves 							= tuning_curves[tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]

occupancy 							= np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]

angle_wak, proba_angle_wak,_		= decodeHD(tcurves, spikes, wake_ep, bin_size = 200, px = occupancy)

angle_sleep, proba_angle_sleep,_	= decodeHD(tcurves, spikes, sleep_ep, bin_size = 20, px = np.ones_like(occupancy))





figure()
for i,n in zip(tcurves,np.arange(tcurves.shape[1])):
	subplot(4,4,n+1, projection = 'polar')
	plot(tcurves[i], label = i+2)
	legend()



# wake
figure()
ax = subplot(211)
plot(angle_wak)
plot(position['ry'])

subplot(212, sharex = ax)
for i,n in enumerate(tcurves.columns):
	plot(spikes[n].restrict(wake_ep).fillna(peaks[n]), '|', markersize = 10)
	plot(position['ry'])


# sws
figure()
for i,n in enumerate(tcurves.columns):
	plot(spikes[n].restrict(sleep_ep).fillna(peaks[n]), '|', markersize = 10)
	plot(angle_sleep)

show()




sys.exit()

good_exs_wake = [nts.IntervalSet(start = [4.96148e+09], end = [4.99755e+09]),
				nts.IntervalSet(start = [3.96667e+09], end = [3.99714e+09]),
				nts.IntervalSet(start = [5.0872e+09], end = [5.13204e+09])
				]

good_exs_rem = [nts.IntervalSet(start = [8.94993e+09], end = [8.96471e+09])]

good_exs_sws = [nts.IntervalSet(start = [8.36988e+09], end = [8.37194e+09])]


figure()
subplot(131)
plot(angle_wak.restrict(good_exs_wake[0]))
plot(position['ry'].restrict(good_exs_wake[0]))
for i,n in enumerate(tcurves.columns):
	plot(spikes[n].restrict(good_exs_wake[0]).fillna(peaks[n]), '|', markersize = 10)
	
ylim(0, 2*np.pi)

subplot(132)
plot(angle_rem.restrict(good_exs_rem[0]))
for i,n in enumerate(tcurves.columns):
	plot(spikes[n].restrict(good_exs_rem[0]).fillna(peaks[n]), '|', markersize = 10)

ylim(0, 2*np.pi)

subplot(133)
plot(angle_sws.restrict(good_exs_sws[0]))
for i,n in enumerate(tcurves.columns):
	plot(spikes[n].restrict(good_exs_sws[0]).fillna(peaks[n]), '|', markersize = 10)

ylim(0, 2*np.pi)

show()



sys.exit()

figure()
ax = subplot(211)
plot(entropy)
plot(filterd)
axvline(ep1.iloc[0,0])
axvline(ep1.iloc[0,1])
axvline(ep2.iloc[0,0])
axvline(ep2.iloc[0,1])
subplot(212, sharex = ax)
plot(acceleration.iloc[:,0])
axvline(ep1.iloc[0,0])
axvline(ep1.iloc[0,1])
axvline(ep2.iloc[0,0])
axvline(ep2.iloc[0,1])

show()







# plot(spikes[7].restrict(ep1).fillna(7).as_units('ms'), '|')
# plot(spikes[8].restrict(ep1).fillna(8).as_units('ms'), '|')
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle


data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

sessions = ['A1407-190416', 'A1407-190417', 'A1407-190422']

for s in sessions[0:1]:
	path = os.path.join(data_directory, s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	episodes 							= info.filter(like='Trial').loc[s].dropna().values
	events								= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sleep_ep 							= loadEpoch(path, 'sleep')					
	sws_ep								= loadEpoch(path, 'sws')
	rem_ep								= loadEpoch(path, 'rem')

	tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
	tcurves 							= smoothAngularTuningCurves(tuning_curves[1], 10, 2)
	tokeep, stat 						= findHDCells(tcurves)
	tcurves 							= tuning_curves[1][tokeep]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
	tcurves 							= tcurves[peaks.index.values]
	# neurons 							= [s+'_'+str(n) for n in tcurves.columns.values]
	# peaks.index							= pd.Index(neurons)
	# tcurves.columns						= pd.Index(neurons)


	occupancy 							= np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]

	sys.exit()

	angle_wak, proba_angle_wak			= decodeHD(tcurves, spikes, wake_ep, bin_size = 200, px = occupancy)

	angle_sleep, proba_angle_sleep		= decodeHD(tcurves, spikes, sleep_ep.loc[[1]], bin_size = 200, px = np.ones_like(occupancy))

	angle_rem 							= angle_sleep.restrict(rem_ep)

	angle_sleep, proba_angle_sleep		= decodeHD(tcurves, spikes, sleep_ep.loc[[1]], bin_size = 50, px = np.ones_like(occupancy))

	angle_sws 							= angle_sleep.restrict(sws_ep)
	


############################################################################
# SAVINGt
############################################################################
datatosave = {	'wak':angle_wak,
				'rem':angle_rem,
				'sws':angle_sws,
				'tcurves':tcurves,
				'angle':position['ry'],
				'peaks':peaks,
				'proba_angle_sws':proba_angle_sleep
			}


cPickle.dump(datatosave, open('../figures/figures_poster_2019/fig_1_decoding.pickle', 'wb'))


############################################################################
# PLOT
############################################################################



figure()
for i,n in zip(tcurves,np.arange(tcurves.shape[1])):
	subplot(3,4,n+1, projection = 'polar')
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

# rem
figure()
for i,n in enumerate(tcurves.columns):
	plot(spikes[n].restrict(rem_ep).fillna(peaks[n]), '|', markersize = 10)
	plot(angle_rem)

show()

# sws
figure()
for i,n in enumerate(tcurves.columns):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)
	plot(angle_sws)

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
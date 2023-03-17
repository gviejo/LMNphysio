import numpy as np
import pandas as pd
import sys
sys.path.append("../")
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle


data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

# s = 'LMN/A1411/A1411-200908A'
s = 'LMN-ADN/A5011/A5011-201014A'


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

# NEURONS FROM ADN	
if 'A5011' in s:
	adn = np.where(shank <=3)[0]
	lmn = np.where(shank ==5)[0]

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
	tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
	tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)
	tokeep, stat = findHDCells(tcurves_half)
	tokeep2.append(tokeep)
	stats2.append(stat)
	tcurves2.append(tcurves_half)

tokeep = np.intersect1d(tokeep2[0], tokeep2[1])


# Checking firing rate
spikes = {n:spikes[n] for n in tokeep}
mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])	
# tokeep = mean_frate[(mean_frate.loc[tokeep]>4).all(1)].index.values
# tokeep = mean_frate[mean_frate.loc[tokeep,'sws']>1].index.values

tcurves 		= tuning_curves[tokeep]
peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()
tcurves 		= tcurves[peaks.index.values]

adn = np.intersect1d(adn, tokeep)
lmn = np.intersect1d(lmn, tokeep)

tokeep = np.hstack((adn, lmn))


# occupancy 							= np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]


# angle_wak, proba_angle_wak,_		= decodeHD(tcurves, spikes, wake_ep.loc[[0]], bin_size = 50, px = occupancy)

# angle_sleep, proba_angle_sleep,_	= decodeHD(tcurves, spikes, sleep_ep.loc[[1]], bin_size = 50, px = np.ones_like(occupancy))

# angle_rem 							= angle_sleep.restrict(rem_ep)

# angle_sleep, proba_angle_sleep, spike_counts	= decodeHD(tcurves, spikes, sleep_ep.loc[[1]], bin_size = 10, px = np.ones_like(occupancy))

# angle_sws 							= angle_sleep.restrict(sws_ep)
	


############################################################################
# SAVINGt
############################################################################
# datatosave = {	'wak':angle_wak,
# 				'rem':angle_rem,
# 				'sws':angle_sws,
# 				'tcurves':tcurves,
# 				'angle':position['ry'],
# 				'peaks':peaks,
# 				'proba_angle_sws':proba_angle_sleep,
# 				'spike_counts':spike_counts

# 			}


# cPickle.dump(datatosave, open('../../figures/figures_ipn_2020/fig_1_decoding.pickle', 'wb'))


############################################################################
# PLOT
############################################################################


decoding = cPickle.load(open('../../figures/figures_poster_2021/fig_cosyne_decoding.pickle', 'rb'))


# figure()
# for i,n in zip(tcurves,np.arange(tcurves.shape[1])):
# 	subplot(3,10,n+1, projection = 'polar')
# 	plot(tcurves[i], label = i+2)
# 	legend()



# wake
figure()
ax = subplot(211)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(wake_ep).fillna(peaks[n]), '|', markersize = 10)
	plot(position['ry'])
subplot(212, sharex = ax)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(wake_ep).fillna(peaks[n]), '|', markersize = 10)
	plot(position['ry'])



# rem
figure()
#ex_rem = nts.IntervalSet(start = 1.57085e+10, end = 1.57449e+10)
ax = subplot(211)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(rem_ep).fillna(peaks[n]), '|', markersize = 10)	
subplot(212, sharex = ax)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(rem_ep).fillna(peaks[n]), '|', markersize = 10)
	


# sws
figure()
ax = subplot(211)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
tmp2 = decoding['sws'].rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
plot(tmp2, '--', linewidth = 2, color = 'black')


subplot(212, sharex = ax)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)
tmp2 = decoding['sws'].rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
plot(tmp2, '--', linewidth = 2, color = 'black')
	


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
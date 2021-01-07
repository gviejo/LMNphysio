import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv

data_directory = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201014A'


episodes = ['sleep', 'wake', 'sleep']
events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
acceleration 						= acceleration[[0,1,2]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)
sws_ep 								= loadEpoch(data_directory, 'sws')
rem_ep 								= loadEpoch(data_directory, 'rem')

############################################################################################### 
# COMPUTING TUNING CURVES
###############################################################################################
tuning_curves, velocity, bins_velocity = computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)


tokeep, stat = findHDCells(tuning_curves[1], z = 20, p = 0.001 , m = 1)

adn = list(np.where(shank <= 3)[0])
lmn = list(np.where(shank == 5)[0])
adn = np.intersect1d(adn, tokeep)
lmn = np.intersect1d(lmn, tokeep)

if 'A5011-201014A' in data_directory:
	adn = adn[:-3]



tcurvesall = {}
peaksall = {}
for neurons, name in zip([adn, lmn], ['adn', 'lmn']):
	tcurves 							= tuning_curves[1][neurons]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
	tcurves 							= tcurves[peaks.index.values]
	tcurvesall[name] 					= tcurves
	peaksall[name]						= peaks


############################################################################
# DECODING
############################################################################
occupancy 							= np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
rawsleep_ep 						= loadEpoch(data_directory, 'sleep')
angle_sleep, proba_angle_sleep,_	= decodeHD(tcurvesall['adn'], spikes, rawsleep_ep.loc[[0]], bin_size = 20, px = np.ones_like(occupancy))

angle_sleep = angle_sleep.restrict(sws_ep)



############################################################################
# PLOT
############################################################################
import matplotlib.gridspec as gridspec

colors = {'adn':'blue', 'lmn':'red'}
count = 1

adn = peaksall['adn'].index.values
lmn = peaksall['lmn'].index.values



fig = figure()
outergs = GridSpec(2,2, figure = fig, width_ratios = [0.15, 0.75], wspace = 0.1)

# TUNING CURVES
gs_tc1 = gridspec.GridSpecFromSubplotSpec(len(adn)//3 + 1,3, subplot_spec = outergs[0,0])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)
for i, n in enumerate(adn[::-1]):
	subplot(gs_tc1[i//3,i%3], projection = 'polar')
	clr = hsluv.hsluv_to_rgb([peaksall['adn'][n]*180/np.pi,85,45])
	plot(tcurvesall['adn'][n], color = clr)	
	xticks([])
	yticks([])
	title(n)
	count += 1
	# gca().spines['top'].set_visible(False)
	# gca().spines['right'].set_visible(False)


gs_tc2 = gridspec.GridSpecFromSubplotSpec(len(lmn)//3 + 1,3, subplot_spec = outergs[1,0])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)
for i, n in enumerate(lmn[::-1]):
	subplot(gs_tc2[i//3,i%3], projection = 'polar')
	clr = hsluv.hsluv_to_rgb([peaksall['lmn'][n]*180/np.pi,85,45])
	plot(tcurvesall['lmn'][n], color = clr)	
	xticks([])
	yticks([])	
	count += 1
	# gca().spines['top'].set_visible(False)
	# gca().spines['right'].set_visible(False)


# SPIKES
# gs_spk = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec = outergs[0,0])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)
# for i, ep in enumerate([wake_ep, rem_ep, sws_ep]):	
ax = subplot(outergs[0,1])
plot(angle_sleep)
for i,n in enumerate(adn):
	clr = hsluv.hsluv_to_rgb([peaksall['adn'][n]*180/np.pi,85,45])
	plot(spikes[n].restrict(sws_ep).fillna(peaksall['adn'][n]), '|', markersize = 10, color = clr)

subplot(outergs[1,1], sharex = ax)	
plot(angle_sleep)
for i,n in enumerate(lmn):
	clr = hsluv.hsluv_to_rgb([peaksall['lmn'][n]*180/np.pi,85,45])
	plot(spikes[n].restrict(sws_ep).fillna(peaksall['lmn'][n]), '|', markersize = 10, color = clr)




# # CROSS-CORRS
# # data = cPickle.load(open('../../figures/figures_poster_2019/fig_2_crosscorr.pickle', 'rb'))
# data = cPickle.load(open('../data/test_lmn_adn_crosscorr.pickle', 'rb'))

# tcurves		 		= data['tcurves']
# pairs 				= data['pairs']
# sess_groups	 		= data['sess_groups']
# frates		 		= data['frates']
# cc_wak		 		= data['cc_wak']
# cc_rem		 		= data['cc_rem']
# cc_sws		 		= data['cc_sws']
# peaks 				= data['peaks']

# fig = figure()
# outergs = GridSpec(1,4, figure = fig)#, width_ratios = [0.25, 0.75], wspace = 0.3)
# # angular differences
# subplot(outergs[0,0])
# # simpleaxis(gca())
# plot(pairs.values, np.arange(len(pairs))[::-1])
# xticks([0, np.pi], ['0', r'$\pi$'], fontsize = 6)
# yticks([0, len(pairs)-1], [len(pairs), 1], fontsize = 6)
# xlabel("Angular\ndifference\n(rad)", fontsize = 7, labelpad = -0.5)
# ylabel("Pairs")
# ylim(0, len(pairs)-1)

# for i, epoch, cc in zip(range(3), ['WAKE', 'REM', 'NREM'], [cc_wak, cc_rem, cc_sws]):
# 	subplot(outergs[0,i+1])
# 	# simpleaxis(gca())
# 	tmp = cc[pairs.index]
# 	tmp = tmp - tmp.mean(0)
# 	tmp = tmp / tmp.std(0)
# 	tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))

# 	imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
# 	times = cc.index.values
# 	xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])], fontsize = 6)	
# 	yticks([0, len(pairs)-1], [1, len(pairs)], fontsize = 6)
# 	title(epoch)
# 	xlabel("Time lag (ms)", fontsize = 7)

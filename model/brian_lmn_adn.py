# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-02-21 12:10:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-11-22 16:33:43

import scipy.io
import sys,os
import numpy as np
from brian2 import *
import pandas as pd
import pynapple as nap
from functions import smoothAngularTuningCurves
import sys
from itertools import combinations, product




############################################################################################### 
# GENERAL infos
###############################################################################################
name = 'A5011-201014A'
path = '/home/guillaume/Dropbox/CosyneData/A5011-201014A'

path2 = '/home/guillaume/Dropbox/CosyneData'


############################################################################################### 
# LOADING DATA
###############################################################################################
data = nap.load_session(path, 'neurosuite')
# data = nap.load_session('/home/guillaume/Dropbox/LMN/A5000/A5011/A5011-201011A', 'neurosuite')

spikes = data.spikes#.getby_threshold('freq', 1.0)
angle = data.position['ry']
position = data.position
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')
rem_ep = data.read_neuroscope_intervals('rem')

# Only taking the first wake ep
wake_ep = wake_ep.loc[[0]]

adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
spikes = spikes.getby_threshold('SI', 0.1, op = '>')
tuning_curves = tuning_curves[spikes.keys()]

tokeep = list(spikes.keys())

adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values

tcurves = tuning_curves

tokeep = np.hstack((adn, lmn))

import _pickle as cPickle
sys.path.append('../python')
import neuroseries as nts

tmp = cPickle.load(open(path2+'/figures_poster_2021/fig_cosyne_decoding.pickle', 'rb'))

decoding = {
	'wak':nap.Tsd(t=tmp['wak'].index.values, d=tmp['wak'].values, time_units = 'us'),
	'sws':nap.Tsd(t=tmp['sws'].index.values, d=tmp['sws'].values, time_units = 'us'),
	'rem':nap.Tsd(t=tmp['rem'].index.values, d=tmp['rem'].values, time_units = 'us'),	
}


# start = 4400.600000
# end = 4402.154216186978

start = 7590.0
end = 7600.0


restrict_ep = nap.IntervalSet(start = start, end = end, time_units = 's')






og_spikes = {}
spk_times = []
for i, n in enumerate(adn):
	og_spikes[i] = nap.Ts(spikes[n].restrict(restrict_ep).index.values - restrict_ep.loc[0, 'start'])
og_spikes = nap.TsGroup(og_spikes)
og_spikes.set_info(peak=tcurves[adn].idxmax().reset_index(drop=True))



lmn_spikes = {}
spk_times = []
for i, n in enumerate(lmn):
	lmn_spikes[i] = nap.Ts(spikes[n].restrict(restrict_ep).index.values - restrict_ep.loc[0, 'start'])
	tmp = spikes[n].restrict(restrict_ep).as_units('ms').index.values
	tmp -= restrict_ep.as_units('ms').loc[0,'start']	
	idx = np.diff(tmp) > 1
	tmp = tmp[0:-1][idx]
	spk_times.append(pd.Series(index = tmp, data = i, dtype = np.int))

lmn_spikes = nap.TsGroup(lmn_spikes)

spk_times = pd.concat(spk_times).sort_index()


new_ep = restrict_ep


#############################################################################
# BRIAN NETWORK
#############################################################################
set_device('cpp_standalone')
start_scope()


duration = new_ep.tot_length() * second

tau_adn = 100 * ms
tau_lmn = 100 * ms

dt_lmn = 1 * ms

n_lmn = len(lmn)

indices = spk_times.values
spk_times = spk_times.index.values * ms

LMN_group = SpikeGeneratorGroup(n_lmn, indices, spk_times, dt=dt_lmn)


eqs_adn = '''
dv/dt = ((10/(1+exp(-(I-0.75)*60)))-v)/tau_adn : 1
dI/dt = -I/tau_adn : 1
'''
# eqs_adn = '''
# dv/dt = -v/ tau-
# '''
eqs_inh 	= '''
dv/dt = - v/tau_lmn : 1
'''

ADN_group = NeuronGroup(n_lmn, model=eqs_adn, threshold='v>0.95', reset='v=0.0', method = 'euler', refractory=10*ms)
INT_group = NeuronGroup(n_lmn, model=eqs_inh, threshold='v>1', reset='v=0.0', method = 'euler', refractory=10*ms)
#INH_group = NeuronGroup(1, model = eqs_inh, threshold='v>1', reset='v=0', method = 'exact')

# ###########################################################################################################
# # SYNAPSES
# ###########################################################################################################
# LMN to ADN connection
LMN_to_ADN = Synapses(LMN_group, ADN_group, 'w : 1', on_pre='I += w')
LMN_to_ADN.connect(i = np.arange(n_lmn), j = np.arange(n_lmn))
LMN_to_ADN.w = 0.2
LMN_to_INT = Synapses(LMN_group, INT_group, 'w : 1', on_pre='v += w')
LMN_to_INT.connect(i = np.arange(n_lmn), j = np.arange(n_lmn))
LMN_to_INT.w = 2.0


# ADN_to_INH = Synapses(ADN_group, INH_group, 'w : 1', on_pre='v += w')
# ADN_to_INH.connect(i = np.arange(n_lmn), j = np.zeros(n_lmn, dtype=np.int))
# ADN_to_INH.w = 0.7
# INH_to_ADN = Synapses(INH_group, ADN_group, 'w : 1', on_pre='v -= w')
# INH_to_ADN.connect(i=np.zeros(n_lmn, dtype=np.int), j=np.arange(n_lmn))
# INH_to_ADN.w = 0.7



###########################################################################################################
# Spike monitor
###########################################################################################################
lmn_mon = SpikeMonitor(LMN_group)
adn_mon = SpikeMonitor(ADN_group)
int_mon = SpikeMonitor(INT_group)


###########################################################################################################
# RUN
###########################################################################################################
run(duration, report = 'text')


##########################################
# ADN neurons
##########################################
adn_spikes = pd.DataFrame(index=adn_mon.t / second, data=np.array(adn_mon.i), columns = ['idx'], dtype = "int")
adn_spikes = {n:nap.Ts(t=adn_spikes[adn_spikes==n].dropna().index.values) for n in range(len(ADN_group))}
adn_spikes = nap.TsGroup(adn_spikes, time_support = nap.IntervalSet(0, duration))

int_spikes = pd.DataFrame(index=int_mon.t / second, data=np.array(int_mon.i), columns = ['idx'], dtype = "int")
int_spikes = {n:nap.Ts(t=int_spikes[int_spikes==n].dropna().index.values) for n in range(len(INT_group))}
int_spikes = nap.TsGroup(int_spikes, time_support = nap.IntervalSet(0, duration))


#########################################
# CROSS-CORR
#########################################
# Joining all together
# simspikes = {}
# location = pd.Series()
# count = 0
# for n in lmn:
# 	simspikes[count] = spikes[n]
# 	location.loc[count] = 'lmn'
# 	count += 1
# for n in adn_spikes.keys():
# 	simspikes[count] = adn_spikes[n]
# 	location.loc[count] = 'adn'
# 	count += 1
# for n in int_spikes.keys():
# 	simspikes[count] = int_spikes[n]
# 	location.loc[count] = 'int'
# 	count += 1

# simspikes = nap.TsGroup(simspikes, time_support = sws_ep.intersect(new_ep), location = location)


# cc = nap.compute_crosscorrelogram(simspikes, 5, 500, time_units='ms')

# lmn_neurons = lmn_neurons.iloc[adn_spikes.keys()]
# adn_pfd = pd.Series(index = adn_spikes.keys(), data = lmn_neurons.values)

# peaks = np.hstack((lmn_neurons.values, lmn_neurons.values, lmn_neurons.values))
# peaks = pd.Series(index = simspikes.keys(), data = peaks)


# new_index = cc.columns
# pairs = pd.DataFrame(index = new_index, columns = ['ang diff', 'struct'])
# for i,j in new_index:
# 		a = peaks[i] - peaks[j]		
# 		pairs.loc[(i,j),'ang diff'] = np.abs(np.arctan2(np.sin(a), np.cos(a)))
# lmn_idx = np.where(simspikes._metadata["location"] == "lmn")[0]
# adn_idx = np.where(simspikes._metadata["location"] == "adn")[0]
# int_idx = np.where(simspikes._metadata["location"] == "int")[0]
# pairs.loc[list(combinations(lmn_idx, 2)),'struct'] = 'lmn-lmn'
# pairs.loc[list(combinations(adn_idx, 2)),'struct'] = 'adn-adn'
# pairs.loc[list(product(lmn_idx, adn_idx)),'struct'] = 'lmn-adn'
# pairs.loc[list(combinations(int_idx, 2)),'struct'] = 'int-int'


# #######################################
# # ISI 
# #######################################
# bins = geomspace(0.001, 100.0, 200)
# isi_adn = {}
# for n in adn_spikes.keys():
# 	spk = adn_spikes[n].index.values
# 	tmp = np.diff(spk)
# 	weights = np.ones_like(tmp)/float(len(tmp))
# 	isi_adn[n]= pd.Series(index=bins[0:-1], data=np.histogram(tmp, bins,weights=weights)[0])
# isi_adn = pd.concat(isi_adn, 1)
# isi_adn = isi_adn.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)


##########################################
# FIGURES
##########################################
from matplotlib.pyplot import *
from matplotlib.colors import hsv_to_rgb

mks = 10
medw = 3
alp = 1

# tmp2 = decoding['sws'].restrict(new_ep)
# tmp3 = pd.Series(index = tmp2.index-new_ep.loc[0,'start'], data = np.unwrap(tmp2.values)).rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
# tmp3 = tmp3%(2*np.pi)
tmp3 = angle.restrict(restrict_ep)
tmp3 = nap.Tsd(t=tmp3.index.values - restrict_ep.loc[0,'start'], d = tmp3.values)

figure()
ax = subplot(211)
for k, n in enumerate(adn_spikes.index):
	spk = adn_spikes[n].index.values
	if len(spk):		
		peak = tcurves[lmn[n]].idxmax()
		clr = hsv_to_rgb([peak/(2*np.pi),0.6,0.6])
		plot(spk, np.ones_like(spk)*peak, 'o', color = clr, markersize = 5, markeredgewidth = 0, alpha = 1)
# for k, n in enumerate(og_spikes.index):
# 	spk = og_spikes[n].index.values
# 	if len(spk):		
# 		peak = og_spikes._metadata.loc[k,'peak']
# 		clr = hsv_to_rgb([peak/(2*np.pi),0.6,0.6])
# 		plot(spk, np.ones_like(spk)*peak, '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.5)
plot(tmp3, linewidth = 2, color = 'gray', alpha = alp)
ylim(0, 2*np.pi)


ax = subplot(212, sharex = ax)
for k, n in enumerate(lmn_spikes.index):
	spk = lmn_spikes[n].index.values
	if len(spk):		
		peak = tcurves[lmn[n]].idxmax()
		clr = hsv_to_rgb([peak/(2*np.pi),0.6,0.6])
		plot(spk, np.ones_like(spk)*peak, '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.5)
plot(tmp3, linewidth = 2, color = 'gray', alpha = alp)
ylim(0, 2*np.pi)

show()

import _pickle as cPickle

datatosave = {'lmn_spikes':lmn_spikes,
	'adn_spikes':adn_spikes,
	'angle':tmp3,
	'peak':tcurves[lmn].idxmax().reset_index(drop=True),
	'ep':restrict_ep
	}

path2 = '/home/guillaume/Dropbox/CosyneData'
cPickle.dump(datatosave, open(os.path.join(path2, 'MODEL_RASTER.pickle'), 'wb'))


sys.exit()


names = ['ADN/ADN', 'LMN/ADN', 'LMN/LMN']
ks = ['adn-adn', 'lmn-adn', 'lmn-lmn']
clrs = ['lightgray', 'gray', 'darkgray']

figure(figsize = (16, 6))

for i, n in enumerate(names):
	subpairs = pairs[pairs['struct']==ks[i]]
	group = subpairs.sort_values(by='ang diff').index.values

	angdiff = pairs.loc[group,'ang diff'].values.astype(np.float32)
	group2 = group[angdiff<np.deg2rad(40)]
	group3 = group[angdiff>np.deg2rad(140)]
	pos2 = np.where(angdiff<np.deg2rad(40))[0]
	pos3 = np.where(angdiff>np.deg2rad(140))[0]
	clrs = ['red', 'green']

	subplot(1, 3, i+1)	

	for j,gr in enumerate([group2, group3]):
		cc2 = cc[gr]		
		cc2 = cc2 - cc2.mean(0)
		cc2 = cc2 / cc2.std(0)
		cc2 = cc2#.loc[-0.2:0.2]
		m  = cc2.mean(1)
		s = cc2.std(1)
		plot(cc2.mean(1), color = clrs[j], linewidth = 3)
		fill_between(cc2.index.values, m - s, m+s, color = clrs[j], alpha = 0.1)
	gca().spines['left'].set_position('center')
	xlabel('cross. corr. (s)')	
	locator_params(axis='y', nbins=3)
	ylabel('z',  y = 0.9, rotation = 0, labelpad = 15)
	title(n)


datatosave = {}


figure()

subplot(121)
for st in ['adn-adn', 'int-int']:
	subpairs = pairs[pairs['struct']==st]
	group = subpairs.sort_values(by='ang diff').index.values

	angdiff = pairs.loc[group,'ang diff'].values.astype(np.float32)
	group2 = group[angdiff<np.deg2rad(40)]
	group3 = group[angdiff>np.deg2rad(140)]
	pos2 = np.where(angdiff<np.deg2rad(40))[0]
	pos3 = np.where(angdiff>np.deg2rad(140))[0]
	clrs = ['red', 'green']
	datatosave[st] = []
	for j,gr in enumerate([group2, group3]):
		cc2 = cc[gr]
		cc2 = cc2.dropna(1)
		if st == 'adn-adn':
			cc2 = cc2.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
		else:
			cc2 = cc2.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)			
		cc2 = cc2 - cc2.mean(0)
		cc2 = cc2 / cc2.std(0)
		cc2 = cc2.loc[-0.2:0.2]
		m  = cc2.mean(1)
		s = cc2.std(1)

		datatosave[st].append( pd.DataFrame.from_dict({"m": m, "s": s}) )
		plot(cc2.mean(1), color = clrs[j], linewidth = 3)
		fill_between(cc2.index.values, m - s, m+s, color = clrs[j], alpha = 0.1)
	gca().spines['left'].set_position('center')
	xlabel('cross. corr. (s)')	
	locator_params(axis='y', nbins=3)
	ylabel('z',  y = 0.9, rotation = 0, labelpad = 15)
	title(n)

subplot(122)


show()

import _pickle as cPickle
path2 = '/home/guillaume/Dropbox/CosyneData'
cPickle.dump(datatosave, open(os.path.join(path2, 'MODEL_CC.pickle'), 'wb'))


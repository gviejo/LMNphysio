# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-02 16:49:49
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-04 16:01:59
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
from functions import *
import sys
from itertools import combinations, product
from umap import UMAP
from matplotlib.pyplot import *
from sklearn.manifold import Isomap

data_directory = '/mnt/DataGuillaume/'

sessions = {
	'adn':np.genfromtxt(
		os.path.join(data_directory,'datasets_ADN.list'), 
		delimiter = '\n', 
		dtype = str, 
		comments = '#'),
	'lmn':np.genfromtxt(
		os.path.join(data_directory,'datasets_LMN.list'), 
		delimiter = '\n',
		dtype = str,
		comments = '#')
}
dvs = {}

alltc_sws = {}
alltc_wak = {}

for st in sessions.keys():
	dvs[st] = {}
	alltc_sws[st] = {}
	alltc_wak[st] = {}

	for se in sessions[st]:
		path = os.path.join(data_directory, se)
		data = nap.load_session(path, 'neurosuite')

		spikes = data.spikes.getby_threshold('freq', 1.0)
		angle = data.position['ry']
		wake_ep = data.epochs['wake']
		sleep_ep = data.epochs['sleep']
		sws_ep = data.read_neuroscope_intervals('sws')

		tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
		tuning_curves = smoothAngularTuningCurves(tuning_curves)

		spikes = spikes.getby_category('location')[st]

		# # COMPUTING TUNING CURVES
		tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
		tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)
		SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
		spikes.set_info(SI=SI)
		spikes = spikes.getby_threshold('SI', 0.1, op = '>')
		tuning_curves = tuning_curves[spikes.keys()]
		
		if len(spikes)>=10:
			print(st, se)
			bin_size_sws = 0.01
			sws_ep = sws_ep.intersect(sleep_ep.loc[[0]])
			sws_angle, proba = nap.decode_1d(tuning_curves=tuning_curves, 
                                     group=spikes,
                                     ep=sws_ep,
                                     bin_size=bin_size_sws,
                                     feature=angle,
                                    )
			sws_angle2 = smoothAngle(sws_angle, 1)
			av_sws = getAngularVelocity(sws_angle2, bin_size_sws)
			logl = nap.Tsd(np.log(proba.max(1).astype(np.float64)), time_support = sws_ep)			
			thr = np.percentile(logl.values, 0.5) -np.random.rand()*1e-7
			logl = logl.threshold(thr)
			av_sws = av_sws.restrict(logl.time_support)
			dvs[st][se] = av_sws
			

			# Tuning curves angular velocity			
			tc_av_sws = nap.compute_1d_tuning_curves(spikes, av_sws, 40, minmax=(0,2*np.pi), ep = av_sws.time_support)			

			bin_size_wake = 0.3
			wak_angle, proba = nap.decode_1d(tuning_curves=tuning_curves, 
                                     group=spikes,
                                     ep=angle.time_support,
                                     bin_size=bin_size_wake,
                                     feature=angle,
                                    )			
			wak_angle2 = smoothAngle(wak_angle, 1)
			av_wak = getAngularVelocity(wak_angle2, bin_size_wake)
			logl = nap.Tsd(np.log(proba.max(1).astype(np.float64)), time_support = wak_angle.time_support)
			thr = np.percentile(logl.values, 0.5) -np.random.rand()*1e-7
			logl = logl.threshold(thr)
			av_wak = av_wak.restrict(logl.time_support)
			tc_av_wak = nap.compute_1d_tuning_curves(spikes, av_wak, 40, minmax = (0, 2*np.pi), ep = av_wak.time_support)

			alltc_wak[st][se] = tc_av_wak
			alltc_sws[st][se] = tc_av_sws


vmax = 0
vmin = np.inf

dvs2 = {}
for gr in dvs.keys():
	dvs2[gr] = []
	for s in dvs[gr].keys():
		tmp = dvs[gr][s].values
		tmp = tmp[~np.isinf(tmp)]
		tmp = tmp[~np.isnan(tmp)]
		dvs2[gr].append(tmp)
		vmax = np.maximum(vmax, np.max(tmp))
		vmin = np.maximum(vmin, np.min(tmp))

# dvlmn = (dvs['lmn'].values)
# dvadn = (dvs['adn'].values)
# dvadn = dvadn[~np.isinf(dvadn)]
# dvlmn = dvlmn[~np.isinf(dvlmn)]

bins = np.linspace(0, vmax/2, 20)
lmn = np.array([np.histogram(a, bins, density=True)[0] for a in dvs2['lmn']]).T
adn = np.array([np.histogram(a, bins, density=True)[0] for a in dvs2['adn']]).T

datatosave = {'adn':adn,'lmn':lmn}

import _pickle as cPickle
cPickle.dump(datatosave, open(os.path.join('../data/', 'All_SWS_dynamic.pickle'), 'wb'))


plot(lmn.mean(1), label = 'lmn')
m = lmn.mean(1)
s = lmn.std(1)
fill_between(range(len(m)), m-s, m+s, alpha = 0.5)
plot(adn.mean(1), label = 'adn')
m = adn.mean(1)
s = adn.std(1)
fill_between(range(len(m)), m-s, m+s, alpha = 0.5)
legend()


## TUNING CURVES
dftc = {}

for ep, alltc in zip(['wak', 'sws'], [alltc_wak, alltc_sws]):
	for st in alltc.keys():
		tmp = [alltc[st][s] for s in alltc[st].keys()]
		tmp = pd.concat(tmp, 1)
		dftc[ep+'_'+st] = tmp.mean(1)

dftc = pd.DataFrame(dftc)

figure()
subplot(121)
for e in ['wak_adn', 'wak_lmn']:
	plot(dftc[e], label = e)
	legend()
subplot(122)
for e in ['sws_adn', 'sws_lmn']:
	plot(dftc[e], label = e)
	legend()
show()
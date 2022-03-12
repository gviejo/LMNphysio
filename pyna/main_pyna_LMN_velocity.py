# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-02 16:41:05
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-03 15:24:35
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-02 12:48:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-02 16:40:45
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

path = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201014A'
data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('freq', 1.0)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')

# # COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI=SI)
spikes = spikes.getby_threshold('SI', 0.1, op = '>')
tuning_curves = tuning_curves[spikes.keys()]
peaks_adn = tuning_curves.idxmax()

# # Decoding
# bin_size = 0.01
# sws_angle, proba = nap.decode_1d(tuning_curves, spikes, sws_ep, bin_size, feature=angle.restrict(angle.time_support.loc[[0]]))
# sws_angle2 = sws_angle.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1)
# sws_angle2 = nap.Tsd(sws_angle2, time_support = sws_ep)
# dv = np.abs(np.diff(np.unwrap(sws_angle2.values)))
# dv = nap.Tsd(t = sws_angle2.index.values[0:-1]+bin_size/2, d = dv, time_support = sws_ep)
# logl = nap.Tsd(proba.max(1), time_support = sws_ep)
# logl = logl.threshold(0.06)
# alpha = dv.restrict(logl.time_support)

# DECODING XGB
# Binning Wake
bin_size_wake = 0.3
count = spikes.count(bin_size_wake, angle.time_support.loc[[0]])
count = count.as_dataframe()
ratewak = np.sqrt(count/bin_size_wake)
ratewak = ratewak.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
ratewak = zscore_rate(ratewak)
velocity = computeLinearVelocity(data.position[['x', 'z']], angle.time_support.loc[[0]], bin_size_wake)
newwake_ep = velocity.threshold(0.001).time_support	
ratewak = ratewak.restrict(newwake_ep)
angle2 = getBinnedAngle(angle, angle.time_support.loc[[0]], bin_size_wake).restrict(newwake_ep)

# Binning sws
bin_size_sws = 0.010
count = spikes.count(bin_size_sws, sws_ep)
count = count.as_dataframe()
ratesws = np.sqrt(count/bin_size_wake)
ratesws = ratesws.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
ratesws = zscore_rate(ratesws)

sws_angle, proba = xgb_decodage(Xr=ratewak, Yr=angle2, Xt=ratesws)
sws_angle2 = sws_angle.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1)
sws_angle2 = nap.Tsd(sws_angle2, time_support = sws_ep)
dv = np.abs(np.diff(np.unwrap(sws_angle2.values)))
dv = nap.Tsd(t = sws_angle2.index.values[0:-1]+bin_size_sws/2, d = dv, time_support = sws_ep)
logl = nap.Tsd(proba.max(1), time_support = sws_ep)
logl = logl.threshold(0.015)
dv = dv.restrict(logl.time_support)




figure()
hist(dv.values, 100)



figure()
ax = subplot(311)
for n in spikes.keys():
    plot(spikes[n].restrict(sws_ep).as_units('s').fillna(peaks_adn.loc[n]), '|')
# plot(sws_angle.restrict(sws_ep).as_units('s'))
plot(sws_angle2.restrict(sws_ep).as_units('s'))
xlabel("Time (s)")
subplot(312, sharex = ax)
plot(logl, '.-')
subplot(313, sharex = ax)
plot(dv, '.-')
show()

np.save(open("lmn.npy", "wb"), dv)


# ############################################################################################### 
# # FIGURES
# ###############################################################################################

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = spikes._metadata['group']

figure()
count = 1
for j in np.unique(shank):
	neurons = shank.index[np.where(shank == j)[0]]
	for k,n in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		plot(tuning_curves[n], label = str(shank.loc[n]) + ' ' + str(n), color = colors[shank.loc[n]-1])
		legend()
		title(spikes._metadata['SI'].loc[n])
		count+=1
		gca().set_xticklabels([])
show()


dvlmn = dv
dvadn = np.load(open('adn.npy', 'rb'))

bins = np.linspace(0, np.maximum(dvadn.max(), dvlmn.max()), 100)
lmn = np.histogram(dvlmn, bins, density=True)[0]
adn = np.histogram(dvadn, bins, density=True)[0]

plot(lmn, label = 'lmn')
plot(adn, label = 'adn')
legend()
show()
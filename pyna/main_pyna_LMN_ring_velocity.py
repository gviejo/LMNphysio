# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 16:35:14
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-13 15:57:12
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

spikes = data.spikes.getby_threshold('rate', 1.0)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)

spikes = spikes.getby_category('location')['lmn']

# # COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
spikes = spikes.getby_threshold('SI', 0.1, op = '>')


# Binning Wake
bin_size_wake = 0.3
count = spikes.count(bin_size_wake, angle.time_support.loc[[0]])
count = count.as_dataframe()
ratewak = np.sqrt(count/bin_size_wake)
ratewak = ratewak.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
ratewak = zscore_rate(ratewak)
ratewak = StandardScaler().fit_transform(ratewak.values)
rgb = getRGB(angle, angle.time_support.loc[[0]], 0.3)
velocity = computeLinearVelocity(data.position[['x', 'z']], angle.time_support.loc[[0]], bin_size_wake)
newwake_ep = velocity.threshold(0.001).time_support	
rgb = rgb.restrict(newwake_ep)
ratewak = ratewak.restrict(newwake_ep)

# Binning sws
bin_size_sws = 0.030
sws_ep =sws_ep.intersect(sleep_ep.loc[[0]])
count = spikes.count(bin_size_sws, sws_ep)
count = count.as_dataframe()
ratesws = np.sqrt(count/bin_size_wake)
ratesws = ratesws.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
mrate = nap.Tsd(ratesws.mean(1), time_support = sws_ep)
newep = mrate.threshold(mrate.quantile(0.25)).time_support
ratesws = nap.TsdFrame(ratesws, time_support = sws_ep)
ratesws = ratesws.restrict(newep)
ratesws = zscore_rate(ratesws)
ratesws = ratesws.iloc[0:10000]

rates = np.vstack([ratesws.values, ratewak.values])

group = np.hstack((np.ones(len(ratesws)), np.zeros(len(ratewak))))

clrs = np.vstack([np.ones((len(ratesws),3))*0.2, rgb.values])



# ump = UMAP(n_neighbors = 50, ).fit_transform(rates)
ump = Isomap(n_neighbors = 40, n_components = 2).fit_transform(rates)

alpha = np.unwrap(np.arctan2(ump[group == 1][:, 1], ump[group == 1][:, 0]))
dt = np.diff(ratesws.index.values)
dv = np.abs(np.diff(alpha))
dv = dv[dt <= bin_size_sws * 2]


fig = figure()
scatter(ump[:,0], ump[:,1], c = clrs, alpha = 0.5)
# ax = fig.add_subplot(1,1,1,projection='3d')
# ax.scatter(ump[:,0], ump[:,1], ump[:,2], c = clrs)

figure()
hist(dv, 100)



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
lmn = np.histogram(dvlmn, bins)[0]
adn = np.histogram(dvadn, bins)[0]

plot(lmn, label = 'lmn')
plot(adn, label = 'adn')
legend()
show()
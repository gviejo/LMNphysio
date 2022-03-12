# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 21:36:00
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-07 10:13:58
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
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.decomposition import KernelPCA
from scipy.ndimage import gaussian_filter

#path = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201010A'
path = '/mnt/DataGuillaume/ADN/Mouse32/Mouse32-140822'
# path = '/mnt/DataGuillaume/ADN/Mouse12/Mouse12-120806'

data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('freq', 1.0)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')
spikes = spikes.getby_category('location')['adn']

# # COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI=SI)
spikes = spikes.getby_threshold('SI', 0.7, op = '>')


# Binning Wake
bin_size_wake = 0.3
count = spikes.count(bin_size_wake, angle.time_support.loc[[0]])
ratewak = np.sqrt(count.as_dataframe()/bin_size_wake)
#ratewak = count.as_dataframe()/bin_size_wake
ratewak = ratewak.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
ratewak = nap.TsdFrame(ratewak, time_support = angle.time_support.loc[[0]])
ratewak = zscore_rate(ratewak)
meanratewak = nap.Tsd(ratewak.max(1))
rgb = getRGB(angle, angle.time_support.loc[[0]], bin_size_wake)
try:
	velocity = computeLinearVelocity(data.position[['x', 'z']], data.position.time_support.loc[[0]], bin_size_wake)
	newwake_ep = velocity.threshold(0.001).time_support.merge_close_intervals(5)
except:
	velocity = computeAngularVelocity(data.position['ry'], data.position.time_support.loc[[0]], bin_size_wake)
	newwake_ep = velocity.threshold(0.01).time_support.merge_close_intervals(5)
rgb = rgb.restrict(newwake_ep)
ratewak = ratewak.restrict(newwake_ep)
meanratewak = meanratewak.restrict(newwake_ep)

# Binning sws
bin_size_sws = 0.02
sws_ep =sws_ep.intersect(sleep_ep.loc[[0]])
count = spikes.count(bin_size_sws, sws_ep)
ratesws = np.sqrt(count.as_dataframe()/bin_size_sws)
#ratesws = count.as_dataframe()/bin_size_sws
ratesws = ratesws.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
ratesws = nap.TsdFrame(ratesws, time_support = sws_ep)
ratesws = zscore_rate(ratesws)
meanratesws = nap.Tsd(ratesws.mean(1), time_support = sws_ep)
newep = meanratesws.threshold(meanratesws.quantile(0.8)).time_support
ratesws = ratesws.restrict(newep)
ratesws = ratesws.iloc[0:10000]

rates = np.vstack([ratesws.values, ratewak.values])

group = np.hstack((np.ones(len(ratesws)), np.zeros(len(ratewak))))

clrs = np.vstack([np.ones((len(ratesws),3))*0.2, rgb.values])



#ump = UMAP(n_neighbors = 500, n_components = 2, metric='cosine', min_dist = 0.6).fit_transform(rates)
#ump = Isomap(n_neighbors = 100, n_components = 2).fit_transform(rates)
ump = KernelPCA(n_components = 2, kernel = 'cosine').fit_transform(rates)

#ump = ump - np.array([-0.163, 0.177])

# Angular velocity manifold
alpha = np.arctan2(ump[:,1], ump[:,0])
alpha = (alpha+2*np.pi)%(2*np.pi)
sws_angle, wak_angle = (alpha[group==1], alpha[group==0])
wak_angle = nap.Tsd(t = ratewak.index.values, d = wak_angle, time_support = newwake_ep)
sws_angle = nap.Tsd(t = ratesws.index.values, d = sws_angle)
av_wak, idx_wak = getAngularVelocity(wak_angle, bin_size_wake)
av_sws, idx_sws = getAngularVelocity(sws_angle, bin_size_sws)

# # Radius manifold
radius = np.sqrt(np.sum(np.power(ump, 2), 1))
r_sws, r_wak = (radius[group==1], radius[group==0])
r_wak = nap.Tsd(t = ratewak.index.values[0:-1][idx_wak], d = r_wak[0:-1][idx_wak], time_support = newwake_ep)
r_sws = nap.Tsd(t = ratesws.index.values[0:-1][idx_sws], d = r_sws[0:-1][idx_sws])

# # # Firign rate
# r_wak = nap.Tsd(t = ratewak.index.values[0:-1][idx_wak], d = meanratewak.values[0:-1][idx_wak], time_support = newwake_ep)
# r_sws = nap.Tsd(t = ratesws.index.values[0:-1][idx_sws], d = meanratesws.values[0:-1][idx_sws])
# radius = np.array([r_wak.max(),r_sws.max()])

figure()
subplot(321)
plot(r_wak.values, av_wak.values, 'o')
subplot(322)
plot(r_sws.values, av_sws.values, 'o')
subplot(323)
avmax = 0.2
Hwak, yedges, xedges = np.histogram2d(av_wak.values, r_wak.values,
	bins=[np.linspace(0, avmax, 30), np.linspace(0, radius.max(), 30)])
imshow(Hwak, origin='lower', cmap = 'jet', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])#, aspect= 'equal')
subplot(324)
Hsws, yedges, xedges = np.histogram2d(av_sws.values, r_sws.values,
	bins=[np.linspace(0, avmax, 30), np.linspace(0, radius.max(), 30)])
imshow(Hsws, origin='lower', cmap = 'jet', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])#, aspect= 'equal')
subplot(325)
imshow(gaussian_filter(Hwak, 1), origin='lower', cmap = 'jet', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])#, aspect= 'equal')
subplot(326)
imshow(gaussian_filter(Hsws, 1), origin='lower', cmap = 'jet', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])#, aspect= 'equal')



figure()
subplot(121)
scatter(ump[:,0], ump[:,1], c = clrs, alpha = 0.2)
plot(np.median(ump, 0)[0],np.median(ump, 0)[1], '+', color = 'red', markersize = 10)
subplot(122)
scatter(ump[group==0,0], ump[group==0,1], c = clrs[group==0], alpha = 0.5)


figure()
subplot(221)
hist(av_wak, 100)
subplot(222)
hist(av_sws, 100)
subplot(223)
hist(r_wak, 100)
subplot(224)
hist(r_sws, 100)


# np.save(open("adn.npy", "wb"), dv)

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


# ump = nap.Tsd(t = ratesws.index.values, d = ump[group==1,0])
# ump10 = ump.threshold(10)
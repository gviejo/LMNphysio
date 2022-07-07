# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 14:23:47
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-07-07 17:39:36
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
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap

path = '/mnt/Data2/LMN-PSB-2/A3019/A3019-220701A'
data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('freq', 1.0)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')
up_ep = data.read_neuroscope_intervals('up')
down_ep = data.read_neuroscope_intervals('down')

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

groups = spikes.getby_category('location')

spikes_lmn = groups['lmn'].getby_threshold('SI', 0.3)
spikes_psb = groups['psb'].getby_threshold('SI', 0.3)

bin_size = 0.3



rates = {}
for g, spk in zip(['psb', 'lmn'], [spikes_psb, spikes_lmn]):
    rates[g] = {}
    for e, ep, bin_size in zip(['wak', 'sws'], [angle.time_support.loc[[0]], sws_ep], [bin_size, 0.03]):
        count = spk.count(bin_size, ep)
        count = count.as_dataframe()
        rate = np.sqrt(count/bin_size)
        rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        rates[g][e] = rate

rgb = getRGB(angle, angle.time_support.loc[[0]], bin_size)




tmp = rates['lmn']['wak'].values
imap = Isomap(n_components=2, n_neighbors=50).fit_transform(tmp[0:20000])

figure()
plot(imap[:,0], imap[:,1])
show()




n=20000
imaps = {}
for g in ['psb', 'lmn']:
    # tmp = np.vstack((rates["psb"]["wak"], rates["psb"]["sws"]))
    tmp= rates[g]['wak'].values[0:n]
    imap = Isomap(n_components=2).fit_transform(tmp)
    imaps[g] = imap



figure()
subplot(121)
scatter(imaps['psb'][:,0], imaps['psb'][:,1], color=rgb.values[0:n])
subplot(122)
scatter(imaps['lmn'][:,0], imaps['lmn'][:,1], color=rgb.values[0:n])
show()




# ############################################################################################### 
# # FIGURES
# ###############################################################################################


figure()
for i, n in enumerate(spikes_lmn.keys()):
    subplot(5,5,i+1,projection='polar')
    plot(tuning_curves[n])

figure()
for i, n in enumerate(spikes_psb.keys()):
    subplot(5,5,i+1,projection='polar')
    plot(tuning_curves[n])

show()
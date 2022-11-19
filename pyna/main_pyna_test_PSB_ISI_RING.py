# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 14:23:47
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-09-29 17:11:06
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
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import Isomap

path = '/mnt/Data2/LMN-PSB-2/A3019/A3019-220630A'
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

spikes = spikes.getby_category('location')['lmn'].getby_threshold('SI', 0.1)

bin_size = 0.05



rates = []
rgbs = []
for i in range(3):
    ep = angle.time_support.loc[[i]]
    count = spikes.count(bin_size, ep)
    count = count.as_dataframe()
    rate = np.sqrt(count/bin_size)
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
    rates.append(rate.values)

    rgb = getRGB(angle, ep, bin_size)
    rgbs.append(rgb.values)



# rates = np.vstack(rates)
# rgbs = np.vstack(rgbs)
# imap = Isomap(n_components=3, n_neighbors=50).fit_transform(rates)

# fig = figure()
# ax = fig.add_subplot(1,1,1,projection='3d')
# ax.scatter(imap[:,0], imap[:,1], imap[:,2], c = rgbs)
# show()

# figure()
# plot(imap[:,0], imap[:,1], 'o', color = rgbs)
# show()
sys.exit()
###############################################################################
# ISI 
###############################################################################
bin_size = 0.005

isis = []
rgbs = []
#for i in range(3):
for i in range(1):
    ep = angle.time_support.loc[[i]]
    tmp = nap.Tsd(t = np.arange(ep.start[0], ep.end[0], bin_size), d = 1)
    tmp2 = []
    for n in spikes.keys():
        spk = spikes[n].restrict(ep)
        isi = nap.Tsd(t = spk.index.values[0:-1]+np.diff(spk.index.values)/2, d=np.diff(spk.index.values))
        idx = tmp.index.get_indexer(isi.index, method="nearest")
        isi_tmp = pd.Series(index = tmp.index.values, data = np.nan)
        isi_tmp.loc[tmp.index.values[idx]] = isi.values
        isi_tmp = isi_tmp.fillna(method='ffill')
        isi_tmp = isi_tmp.fillna(0)
        isi_tmp = 1 / (np.log(isi_tmp + 1))        
        isi_tmp = isi_tmp.rolling(window=20,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
        isi_tmp = isi_tmp.fillna(0)        
        isi_tmp = nap.Tsd(isi_tmp)
        tmp2.append(isi_tmp.values)
    tmp2 = np.array(tmp2).T
    isis.append(tmp2)

isis = np.vstack(isis)
isis[np.isnan(isis)] = 0


#imap = KernelPCA(n_components=2, kernel="cosine").fit_transform(isis)

imap = Isomap(n_components=2, n_neighbors=50).fit_transform(isis[0:30000])



figure()
plot(imap[:,0], imap[:,1], 'o')
show()



fig = figure()
ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(imap[:,0], imap[:,1], imap[:,2])
show()

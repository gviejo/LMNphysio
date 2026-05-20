# %%
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 16:35:14
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-11-29 17:01:26
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from scipy.ndimage import gaussian_filter
from umap import UMAP

from functions import *
import sys
from itertools import combinations, product
from umap import UMAP
import matplotlib
# matplotlib.use('TkAgg')  # or 'Agg' if no GUI is needed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import Isomap, MDS
from sklearn.decomposition import PCA, KernelPCA
import warnings

import pymde
import torch


warnings.filterwarnings("ignore")

if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'

path = os.path.join(data_directory, 'LMN-ADN/A5011/A5011-201014A')
# path = os.path.join(data_directory, 'LMN-ADN/A5043/A5043-230301A')

basename = os.path.basename(path)
data = nap.load_file(path + "/kilosort4/" + basename+".nwb")

spikes = data['units']
angle = data['ry']
epochs = data['epochs']
wake_ep = epochs[epochs.tags=="wake"]
sleep_ep = epochs[epochs.tags=="sleep"]
sws_ep = data['sws']
rem_ep = data['rem']

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)

SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

spikes = spikes[spikes.SI>0.1]

# CHECKING HALF EPOCHS
wake2_ep = splitWake(angle.time_support.loc[[0]])
tokeep2 = []
stats2 = []
tcurves2 = []
for i in range(2):
    tcurves_half = nap.compute_1d_tuning_curves(
        spikes, angle, 120, minmax=(0, 2*np.pi),
        ep = wake2_ep[i]
        )
    tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

    tokeep, stat = findHDCells(tcurves_half)
    tokeep2.append(tokeep)
    stats2.append(stat)
    tcurves2.append(tcurves_half)
tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

spikes = spikes[tokeep]

groups = spikes.getby_category('location')

groups = {
    'lmn': groups['lmn'],
    'adn': groups['adn']
}


X_ = {}
X_exs = {}
models = {}

bin_sizes = {
    "wake": 0.2,
    "rem": 0.2,
    "sws": 0.005
}

exs = {
    "wake": nap.IntervalSet(9604.5, 9613.7),
    "sws": nap.IntervalSet(13876, 13880.5),
    "rem": nap.IntervalSet(15710.1, 15720.4)
    }

#%%
# --------------------------
# Binning, smoothing, normalization
# --------------------------
for name, epochs in zip(
        [
            'wake',
            'sws',
            'rem'
        ],
        [
            angle.time_support,
            sws_ep,
            rem_ep
        ]):

    X_[name] = {}
    X_exs[name] = {}

    for loc in groups.keys():

        # X = groups[loc].count(bin_sizes[name], epochs)
        X = np.sqrt(groups[loc].count(bin_sizes[name], epochs))
        # X = np.log(1+groups[loc].count(bin_sizes[name], epochs))
        X = X.smooth(bin_sizes[name]*6, norm=True)
        # X = X.convolve(np.ones(5)/5)
        X = X - X.mean(0)
        X = X / X.std(0)
        # X = X/X.max(0)
        Xex = X.restrict(exs[name])
        # Dropping empty bins for sleep
        if name == 'sws':
            # threshold = np.percentile(X.sum(1), 90)
            # X = X[X.sum(1)>threshold]
            print(np.percentile(X.mean(1), 95))
            thr = np.percentile(X.mean(1), 95)
            X = X[X.mean(1) > thr]
            # Xex = Xex[Xex.mean(1) > thr]
        X_[name][loc] = X
        X_exs[name][loc] = Xex

# %%
# --------------------------
# Dimensionality reduction
# --------------------------

proj = {"wake": {}, "sws": {}, "rem": {}}
proj_ex = {"wake": {}, "sws": {}, "rem": {}}

for loc in ["adn", "lmn"]:


    for name in ['wake', 'rem', 'sws']:
        model = KernelPCA(n_components=2, kernel='cosine')
        # model = Isomap(n_components=2, n_neighbors=200)

        model.fit(X_[name][loc])
        proj[name][loc] = model.transform(X_[name][loc].values)
        proj_ex[name][loc] = model.transform(X_exs[name][loc].values)



colors = getRGB(angle, angle.time_support.loc[[0]], bin_size=bin_sizes['wake'])



#%%


# # --------------------------
# # Tuning curves LMN and ADN
# # --------------------------
# fig, axes = plt.subplots(6, 6, figsize=(12, 12), subplot_kw={'projection': 'polar'})
# axes = axes.flatten()
# count = 0
# for i, loc in enumerate(['adn', 'lmn']):
#     for j, n in enumerate(groups[loc].index):
#         ax = axes[count]
#         color = 'b' if loc == 'lmn' else 'r'
#         ax.plot(tuning_curves[n], color=color)
#         count += 1
#
# # plt.suptitle("Tuning Curves LMN and ADN")
# # plt.tight_layout()
# # plt.show()
#%%
# --------------------------
# Scatter plots of projections
# --------------------------
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(2, 3, wspace=0.3, hspace=0.3)

for j, name in enumerate(['wake', 'rem', 'sws']):
    for i, loc in enumerate(['adn', 'lmn']):
        ax = fig.add_subplot(gs[i, j])
        Y = proj[name][loc]

        if name == 'wake':
            ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=1, alpha=0.7)
        else:
            ax.scatter(Y[:, 0], Y[:, 1], s=1, alpha=0.7)

        ax.plot(
            proj_ex[name][loc][:, 0],
            proj_ex[name][loc][:, 1],
            'o',
            color='black',
            markersize=2,
            markeredgecolor='white',
            markeredgewidth=0.5,
        )

        ax.set_title(f"{loc} {name}")

plt.suptitle("Projections Scatter Plots")
# plt.savefig(os.path.expanduser("~/Dropbox/LMNphysio/data/projections_scatter_KPCA_v1.png"), dpi=300)


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
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import kneighbors_graph
from sklearn.utils._random import sample_without_replacement
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
    "sws": 0.02
}

exs = {
    "wake": nap.IntervalSet(9604.5, 9613.7),
    "sws": nap.IntervalSet(13876, 13880.5),
    "rem": nap.IntervalSet(15710.1, 15720.4)
    }


# %%
# --------------------------
# Dimensionality reduction
# --------------------------
def get_X(data, bin_size, epochs, q_threshold=None):
    X = data.count(bin_size, epochs)
    X = np.sqrt(X.smooth(bin_size*4, norm=False))
    X = X - X.mean(0)
    X = X / X.std(0)
    if q_threshold is not None:
        thr = np.percentile(X.mean(1), q_threshold)
        X = X[X.mean(1) > thr]

    # Y = PCA(n_components=3).fit_transform(X.values)
    # K = diffusion_kernel(X.values, n_neighbors=15, t=1.0)
    # return nap.TsdFrame(t=X.index, d=K, time_support=X.time_support)
    return X

proj = {"wake": {}, "sws": {}, "rem": {}}
proj_ex = {"wake": {}, "sws": {}, "rem": {}}

for loc in ["adn", "lmn"]:

    # Wake + REM + SWS
    allX = {}
    for name, epochs, q_thr in zip(
            ['wake', 'rem', 'sws'],
            [angle.time_support,
             rem_ep.union(angle.time_support),
             sws_ep.union(nap.IntervalSet(angle.time_support.start[0], angle.time_support.start[0] + 60 * 10))],
            [None, None, 95]):

        X = get_X(groups[loc], bin_sizes[name], epochs, q_threshold=q_thr)

        allX[name] = X

    X = []
    for name in ['wake', 'rem', 'sws']:
        # Sample 10000 points for fitting
        n_samples = min(10000, allX[name].shape[0])
        sample_indices = sample_without_replacement(allX[name].shape[0], n_samples, random_state=42)
        X_sampled = allX[name].values[sample_indices]

        X.append(X_sampled)

    X = np.vstack(X)

    model = KernelPCA(n_components=2, kernel='cosine')
    model.fit(X)

    # TRANSFORMATIONS
    for name in ['wake', 'rem', 'sws']:
        X = allX[name]
        proj[name][loc] = model.transform(X.values)
        proj_ex[name][loc] = model.transform(X.restrict(exs[name]).values)

    #
    # # Wake
    # X = get_X(groups[loc], bin_sizes['wake'], angle.time_support)
    #
    # model = KernelPCA(n_components=2, kernel='cosine')
    # model.fit(X.values)
    # proj['wake'][loc] = model.transform(X.values)
    # proj_ex['wake'][loc] = model.transform(X.restrict(exs['wake']).values)
    #
    # # Wake + REM
    # X = get_X(groups[loc], bin_sizes['rem'], rem_ep.union(angle.time_support))
    # model = KernelPCA(n_components=2, kernel='cosine')
    # model.fit(X.values)
    # proj['rem'][loc] = model.transform(X.values)
    # proj_ex['rem'][loc] = model.transform(X.restrict(exs['rem']).values)
    #
    # # Wake + SWS
    # ep = sws_ep.union(nap.IntervalSet(angle.time_support.start[0], angle.time_support.start[0] + 60 * 10))
    # X = get_X(groups[loc], bin_sizes['sws'], ep, q_threshold=95)
    # model = KernelPCA(n_components=2, kernel='cosine')
    # model.fit(X.values)
    # proj['sws'][loc] = model.transform(X.values)
    # proj_ex['sws'][loc] = model.transform(X.restrict(exs['sws']).values)





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
            markersize=5,
            markeredgecolor='white',
            markeredgewidth=0.5,
        )

        ax.set_title(f"{loc} {name}")

plt.suptitle("Projections Scatter Plots")
plt.savefig(os.path.expanduser("~/Dropbox/LMNphysio/data/projections_scatter_KPCA_v3.png"), dpi=300)

#%%
# --------------------------
# Histogram (2D) of projections
# --------------------------
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(2, 3, wspace=0.3, hspace=0.3)

for j, name in enumerate(['wake', 'rem', 'sws']):
    for i, loc in enumerate(['adn', 'lmn']):
        ax = fig.add_subplot(gs[i, j])
        Y = proj[name][loc]
        x = Y[:, 0]
        y = Y[:, 1]
        bins = [np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50)]
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
        Hsmooth = gaussian_filter(H, sigma=2.0)

        img = ax.imshow(
            Hsmooth.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap="turbo"
        )
        ax.plot(
            proj_ex[name][loc][:, 0],
            proj_ex[name][loc][:, 1],
            'o',
            color='black',
            markersize=5,
            markeredgecolor='white',
            markeredgewidth=0.5,
        )

        plt.colorbar(img, ax=ax)
        ax.set_title(f"{loc} {name}")

plt.suptitle("Histogram of Projections")
plt.savefig(os.path.expanduser("~/Dropbox/LMNphysio/data/projections_hist_KPCA_v3.png"), dpi=300)
plt.show()

#%%
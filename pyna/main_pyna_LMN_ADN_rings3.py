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
import _pickle as cPickle


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


datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')



# results will hold projections and colors per session basename
results = {}

# Load sessions from datasets
sessions = datasets

# iterate sessions and compute projections/colors like before
for sess in sessions:
    path = os.path.join(data_directory, sess)
    basename = os.path.basename(path)
    data = nap.load_file(path + "/kilosort4/" + basename+".nwb")

    spikes = data['units']
    angle = data['ry']
    epochs = data['epochs']
    wake_ep = epochs[epochs.tags=="wak"]
    sleep_ep = epochs[epochs.tags=="sleep"]
    sws_ep = data['sws']
    rem_ep = data['rem']

    x = data['x']
    z = data['z']
    position = nap.TsdFrame(
        t=x.index,
        d=np.vstack((x.values, z.values)).T,
        columns=['x', 'z'],
        time_support=x.time_support
    )

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.004).time_support


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
    if (len(groups['lmn'])>5) and (len(groups['adn'])>5):
        X_ = {}
        X_exs = {}
        models = {}

        bin_sizes = {
            "wak": 0.3,
            "rem": 0.3,
            "sws": 0.02
        }

        # --------------------------
        # Binning, smoothing, normalization
        # --------------------------
        for name, epochs_loop in zip(
                [
                    'wak',
                    'sws',
                    'rem'
                ],
                [
                    newwake_ep,
                    sws_ep.intersect(sleep_ep[0]),
                    rem_ep
                ]):

            X_[name] = {}
            # X_exs[name] = {}

            for loc in groups.keys():

                X = groups[loc].count(bin_sizes[name], epochs)
                # X = np.sqrt(groups[loc].count(bin_sizes[name], epochs))
                # X = np.log(1+groups[loc].count(bin_sizes[name], epochs))
                X = X / X.max(0)
                X = X.smooth(bin_sizes[name] * 3, norm=False)
                # X = X - X.mean(0)
                # X = X / X.std(0)
                # X = X/X.max(0)
                # Dropping empty bins for sleep
                if name == 'sws' or name == 'rem':
                    # threshold = np.percentile(X.sum(1), 90)
                    # X = X[X.sum(1)>threshold]
                    # print(np.percentile(X.mean(1), 50))
                    thr = np.percentile(X.mean(1), 50)
                    X = X[X.mean(1) > thr]
                    # Xex = Xex[Xex.mean(1) > thr]

                X = X - X.mean(0)
                X = X / X.std(0)

                X_[name][loc] = X
                # X_exs[name][loc] = X.restrict(exs[name])

        # --------------------------
        # Dimensionality reduction
        # --------------------------

        proj = {"wak": {}, "sws": {}, "rem": {}}

        for loc in ["adn", "lmn"]:

            model = KernelPCA(n_components=2, kernel='cosine')

            # fit on wak activity
            tmp = X_['wak'][loc].values
            model.fit(tmp)

            # transform all epochs
            for name in ['wak', 'sws', 'rem']:
                proj[name][loc] = model.transform(X_[name][loc])


        colors = getRGB(angle, newwake_ep, bin_size=bin_sizes['wak'])

        # store session results
        results[basename] = {
            'proj': proj,
            'colors': colors,
        }


# # Save data
# datatosave = {
#     'proj': proj,
#     'proj_ex': proj_ex,
#     'colors': colors
# }
# dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"
# cPickle.dump(datatosave, open(os.path.join(dropbox_path, "projections_KPCA_LMN_ADN_v2.pickle"), "wb"))
#
# #%%


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
# # --------------------------
# # Scatter plots of projections (combined sessions)
# # --------------------------
session_names = list(results.keys())
n_sessions = len(session_names)
# fig = plt.figure(figsize=(5, 50))
# gs = GridSpec(2, 3 * n_sessions, wspace=0.3, hspace=0.3)
#
# for s_idx, sname in enumerate(session_names):
#     proj = results[sname]['proj']
#     colors = results[sname]['colors']
#     for j, name in enumerate(['wak', 'rem', 'sws']):
#         for i, loc in enumerate(['adn', 'lmn']):
#             ax = fig.add_subplot(gs[i, j + s_idx * 3])
#             Y = proj[name][loc]
#
#             if name == 'wak':
#                 ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=1, alpha=0.7)
#             else:
#                 ax.scatter(Y[:, 0], Y[:, 1], s=1, alpha=0.7)
#
#             ax.set_title(f"{sname} {loc} {name}")
#
# plt.suptitle("Projections Scatter Plots (combined sessions)")
# plt.savefig("/mnt/home/gviejo/Dropbox/LMNphysio/data/projections_scatter_KPCA_all.pdf", dpi=100, bbox_inches='tight')
# plt.close()
#
# #%%
# --------------------------
# Histogram (2D) of projections (combined sessions)
# --------------------------
session_names = list(results.keys())
fig = plt.figure(figsize=(10, 100))
# Outer GridSpec: one row per session with spacing between sessions
gs_outer = GridSpec(n_sessions, 1, hspace=0.5, figure=fig)

for s_idx, sname in enumerate(session_names):
    # Inner GridSpec: 2x3 grid for each session with tight row spacing
    gs_inner = gs_outer[s_idx].subgridspec(2, 3, wspace=0.3, hspace=0.1)

    proj = results[sname]['proj']
    for j, name in enumerate(['wak', 'rem', 'sws']):
        for i, loc in enumerate(['adn', 'lmn']):
            ax = fig.add_subplot(gs_inner[i, j])
            Y = proj[name][loc]
            x = Y[:, 0]
            y = Y[:, 1]
            bins = [np.linspace(-1.2, 1.2, 50), np.linspace(-1.2, 1.2, 50)]
            H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
            H = np.log(1 + H)
            Hsmooth = gaussian_filter(H, sigma=2.0)

            img = ax.imshow(
                Hsmooth.T,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect="equal",
                cmap="turbo"
            )

            # plt.colorbar(img, ax=ax)
            ax.set_title(f"{sname} {loc} {name}")

plt.suptitle("Histogram of Projections (combined sessions)")
plt.savefig("/mnt/home/gviejo/Dropbox/LMNphysio/data/projections_histogram_KPCA_all.pdf", dpi=100, bbox_inches='tight')
plt.close()

#%%
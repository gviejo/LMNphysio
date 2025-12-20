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
import umap
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
from sklearn.preprocessing import StandardScaler
import warnings
import _pickle as cPickle
import hsluv

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
# path = os.path.join(data_directory, 'LMN-ADN/A5044/A5044-240404A')


basename = os.path.basename(path)
data = nap.load_file(path + "/kilosort4/" + basename+".nwb")


spikes = data['units']
angle = data['ry']
epochs = data['epochs']
wake_ep = epochs[epochs.tags=="wak"]
sleep_ep = epochs[epochs.tags=="sleep"]
sws_ep = data['sws']
rem_ep = data['rem']

sws_ep = sws_ep.intersect(sleep_ep[1])

x = data['x']
z = data['z']
position = nap.TsdFrame(
    t=x.index,
    d=np.vstack((x.values, z.values)).T,
    columns=['x', 'z'],
    time_support=x.time_support
)

velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
newwake_ep = velocity.threshold(0.005).time_support

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)

SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

adn_idx = spikes.SI[spikes.location == "adn"] > 0.4
adn_idx = adn_idx[adn_idx].index

# print(adn_idx)
# adn_idx = np.delete(adn_idx, 13)
# print(adn_idx)


lmn_idx = spikes.SI[spikes.location == "lmn"] > 0.01
lmn_idx = lmn_idx[lmn_idx].index
tokeep = np.hstack((adn_idx, lmn_idx))

spikes = spikes[tokeep]

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
idxs = {}

bin_sizes = {
    "wak": 0.3,
    "rem": 0.1,
    "sws": 0.02
}

filepath = os.path.join(os.path.expanduser("~"), f'Dropbox/LMNphysio/data/DATA_FIG_LMN_ADN_{basename}.pickle')
# exdata = cPickle.load(open(filepath, 'rb'))
# exs = {"wak": exdata["ex_wak"], "rem": exdata["ex_rem"], "sws": exdata["ex_sws"]}

#%%
# --------------------------
# Binning, smoothing, normalization
# --------------------------
for name, epochs in zip(
        [
            'wak',
            'sws',
            'rem'
        ],
        [
            newwake_ep,
            sws_ep,
            rem_ep
        ]):

    X_[name] = {}
    # X_exs[name] = {}
    idxs[name] = {}

    for loc in groups.keys():

        # X = groups[loc].count(bin_sizes[name], epochs)
        X = np.sqrt(groups[loc].count(bin_sizes[name], epochs))



        # X = np.log(1+groups[loc].count(bin_sizes[name], epochs))
        # X = X / X.max(0)
        X = X.smooth(bin_sizes[name]*3, norm=False)


        X = X - X.mean(0)
        X = X / X.std(0)

        X_[name][loc] = X
        # X_exs[name][loc] = X.restrict(exs[name])

# Selecting same indices for lmn and adn
thr = np.percentile(X_['sws']['adn'].mean(1), 50)
idxs['sws']['adn'] = (X_['sws']['adn'].mean(1) > thr).values
idxs['sws']['lmn'] = idxs['sws']['adn']

X_['sws']['adn'] = X_['sws']['adn'][idxs['sws']['adn']]
X_['sws']['lmn'] = X_['sws']['lmn'][idxs['sws']['lmn']]

# %%
# --------------------------
# Dimensionality reduction
# --------------------------

proj = {"wak": {}, "sws": {}, "rem": {}}
proj_ex = {"wak": {}, "sws": {}, "rem": {}}

for loc in ["adn", "lmn"]:

    model = KernelPCA(n_components=2, kernel='cosine')
    # model = Isomap(n_components=2, n_neighbors=50, path_method="D", n_jobs=-1)
    #
    # model = umap.UMAP(
    #     n_components=2,  # number of dimensions
    #     metric='cosine',  # distance metric
    #     # options: 'euclidean', 'cosine', 'correlation', etc.
    #     n_neighbors=100,  # local neighborhood size
    #     # larger = more global structure
    #     # smaller = more local structure
    #     min_dist=0.5,  # minimum distance between points in embedding
    #     # smaller = tighter clusters
    #     # larger = more spread out
    #     n_epochs=200,  # number of training epochs (default: auto)
    #     random_state=42,  # reproducibility
    #     n_jobs=-1  # parallel processing
    # )


    tmp = X_['wak'][loc].values
    
    model.fit(tmp)
    models['wak'] = model



    # Sws & Rem
    for name in ['wak', 'sws', 'rem']:
        tmp = X_[name][loc].values
        proj[name][loc] = model.transform(tmp)
        # proj_ex[name][loc] = model.transform(X_exs[name][loc])

# --------------------------
# Decoding
# --------------------------

decoded_color = {}

tuning_curves_adn = nap.compute_tuning_curves(
        spikes[spikes.location=='adn'],
        angle,
        bins=120,
        range=(0, 2*np.pi)
)
for name, epochs in zip(
        [
            'wak',
            'sws',
            'rem'
        ],
        [
            newwake_ep,
            sws_ep,
            rem_ep
        ]):
    decoded, P = nap.decode_bayes(
        tuning_curves=tuning_curves_adn,
        data=spikes[spikes.location=="adn"].restrict(epochs),
        epochs=epochs,
        bin_size=bin_sizes[name],
        sliding_window_size=4,
    )
    decoded_color[name] = {}
    for loc in ["adn", "lmn"]:
        if name == 'sws':
            decoded_color[name][loc] = getRGB(decoded, epochs, bin_size=bin_sizes[name])[idxs[name][loc]]
        else:
            decoded_color[name][loc] = getRGB(decoded, epochs, bin_size=bin_sizes[name])


# colors = {"wak": getRGB(angle, newwake_ep, bin_size=bin_sizes['wak']),
#           "sws": decoded_color["sws"],
#             "rem": decoded_color["rem"]
#             }
colors = decoded_color



# Save data
datatosave = {
    'proj': proj,
    # 'proj_ex': proj_ex,
    'colors': colors
}
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"
cPickle.dump(datatosave, open(os.path.join(dropbox_path, "projections_KPCA_LMN_ADN_v2.pickle"), "wb"))

#%%


# --------------------------
# Tuning curves LMN and ADN
# --------------------------
fig, axes = plt.subplots(5, 6, figsize=(12, 12), subplot_kw={'projection': 'polar'})
axes = axes.flatten()
count = 0
for i, loc in enumerate(['adn', 'lmn']):
    for j, n in enumerate(groups[loc].index):
        ax = axes[count]
        color = 'b' if loc == 'lmn' else 'r'
        h = np.rad2deg(tuning_curves[n].idxmax())
        color = hsluv.hsluv_to_rgb([h, 100, 50])
        ax.plot(tuning_curves[n], color=color)
        ax.fill_between(
            tuning_curves[n].index,
            0,
            tuning_curves[n].values,
            color=color,
            alpha=0.5
        )
        count += 1
        ax.set_title(loc + f" {j} {spikes.SI[n].round(2)}")
        ax.set_xticks([])
        ax.set_yticks([])

# plt.suptitle("Tuning Curves LMN and ADN")
# plt.tight_layout()
# plt.show()
#%%
# --------------------------
# Scatter plots of projections
# --------------------------
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(2, 3, wspace=0.3, hspace=0.3)

for j, name in enumerate(['wak', 'rem', 'sws']):
    for i, loc in enumerate(['adn', 'lmn']):
        ax = fig.add_subplot(gs[i, j])
        Y = proj[name][loc]


        ax.scatter(Y[:, 0], Y[:, 1], c=colors[name][loc], s=0.1, alpha=0.7)

        # ax.plot(
        #     proj_ex[name][loc][:, 0],
        #     proj_ex[name][loc][:, 1],
        #     'o',
        #     color='black',
        #     markersize=5,
        #     markeredgecolor='white',
        #     markeredgewidth=0.5,
        # )

        ax.set_title(f"{loc} {name}")

plt.suptitle("Projections Scatter Plots")
plt.savefig(os.path.expanduser("~/Dropbox/LMNphysio/data/projections_scatter_KPCA_v2.png"), dpi=300)


#%%
# --------------------------
# Histogram (2D) of projections
# --------------------------
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(2, 3, wspace=0.3, hspace=0.3)

for j, name in enumerate(['wak', 'rem', 'sws']):
    for i, loc in enumerate(['adn', 'lmn']):
        ax = fig.add_subplot(gs[i, j])
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
        # ax.plot(
        #     proj_ex[name][loc][:, 0],
        #     proj_ex[name][loc][:, 1],
        #     'o',
        #     color='black',
        #     markersize=5,
        #     markeredgecolor='white',
        #     markeredgewidth=0.5,
        # )

        plt.colorbar(img, ax=ax)
        ax.set_title(f"{loc} {name}")

plt.suptitle("Histogram of Projections")
plt.savefig(os.path.expanduser("~/Dropbox/LMNphysio/data/projections_hist_KPCA_v2.png"), dpi=300)
plt.show()

#%%
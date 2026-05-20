# %%
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2026-04-09
# Comparison of KernelPCA vs Isomap projections for LMN and ADN

import sys, os
import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
import warnings

sys.path.insert(0, '..')
from functions import *

warnings.filterwarnings("ignore")

# %%
# --------------------------
# Data loading
# --------------------------
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

basename = os.path.basename(path)
data = nap.load_file(path + "/kilosort4/" + basename + ".nwb")

spikes = data['units']
angle = data['ry']
epochs = data['epochs']
wake_ep = epochs[epochs.tags == "wake"]
sleep_ep = epochs[epochs.tags == "sleep"]
sws_ep = data['sws']
rem_ep = data['rem']

# %%
# --------------------------
# Tuning curves and cell selection
# --------------------------
tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2 * np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)

SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0, 2 * np.pi))
spikes.set_info(SI)
spikes = spikes[spikes.SI > 0.1]

wake2_ep = splitWake(angle.time_support.loc[[0]])
tokeep2 = []
for i in range(2):
    tcurves_half = nap.compute_1d_tuning_curves(
        spikes, angle, 120, minmax=(0, 2 * np.pi), ep=wake2_ep[i]
    )
    tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)
    tokeep, stat = findHDCells(tcurves_half)
    tokeep2.append(tokeep)
tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
spikes = spikes[tokeep]

groups = spikes.getby_category('location')
groups = {'lmn': groups['lmn'], 'adn': groups['adn']}

# %%
# --------------------------
# Binning, smoothing, normalization
# --------------------------
bin_sizes = {"wake": 0.2, "rem": 0.2, "sws": 0.005}



X_ = {}


for name, ep in zip(['wake', 'sws'], [angle.time_support, sws_ep]):
    X_[name] = {}
    for loc in groups:
        X = np.sqrt(groups[loc].count(bin_sizes[name], ep))
        X = X.smooth(bin_sizes[name] * 6, norm=True)
        X = X - X.mean(0)
        X = X / X.std(0)
        if name == 'sws':
            thr = np.percentile(X.mean(1), 95)
            X = X[X.mean(1) > thr]
        X_[name][loc] = X

# %%
# --------------------------
# Projections: KernelPCA and Isomap
# --------------------------
methods = {
    'KernelPCA': lambda: KernelPCA(n_components=2, kernel='cosine'),
    'Isomap':    lambda: Isomap(n_components=2, n_neighbors=20),
}

proj = {m: {"wake": {}, "sws": {}} for m in methods}

for method_name, make_model in methods.items():
    for loc in ['adn', 'lmn']:
        for name in ['wake']:
            model = make_model()
            model.fit(X_[name][loc])
            proj[method_name][name][loc] = model.transform(X_[name][loc].values)

colors = getRGB(angle, angle.time_support.loc[[0]], bin_size=bin_sizes['wake'])

# %%
# --------------------------
# Figure: KernelPCA (top half) vs Isomap (bottom half)
# Rows: method × region  |  Cols: epoch
# --------------------------
epoch_names = ['wake']
loc_names = ['adn', 'lmn']
method_names = list(methods.keys())  # ['KernelPCA', 'Isomap']

# n_rows = len(method_names) * len(loc_names)  # 4
# n_cols = len(epoch_names)
n_rows = 2
n_cols = 2

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(n_rows, n_cols, wspace=0.3, hspace=0.45)

row_labels = [f"{m}\n{loc}" for m in method_names for loc in loc_names]

name = 'wake'
for row_idx, method_name in enumerate(method_names):
    for col_idx, loc in enumerate(loc_names):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        Y = proj[method_name][name][loc]


        ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=1, alpha=0.7)

        ax.set_title(loc, fontsize=10, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

# Horizontal separator between the two methods
for row_idx, method_name in enumerate(method_names):
    fig.text(0.5, [0.9, 0.5][row_idx], f'── {method_name} ──', ha='center', va='center',
         fontsize=12, color='black', fontweight='bold')

plt.suptitle("KernelPCA vs Isomap — LMN / ADN projections", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/manifold_comparison_KPCA_vs_Isomap.png"),
    dpi=300, bbox_inches='tight'
)
# plt.show()
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
import torch
import torch.nn as nn


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

# path = os.path.join(data_directory, 'LMN-ADN/A5011/A5011-201014A')
path = os.path.join(data_directory, 'LMN-ADN/A5043/A5043-230301A')

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
newwake_ep = velocity.threshold(0.001).time_support

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
    "wak": 0.2,
    "rem": 0.2,
    "sws": 0.005
}

# exs = {
#     "wak": nap.IntervalSet(9604.5, 9613.7),
#     "sws": nap.IntervalSet(start = 13876.0, end=13877.5, time_units = 's'),
#     "rem": nap.IntervalSet(15710.1, 15720.4)
#     }

# exs = {
#     'wak':nap.IntervalSet(angle.time_support.start[0]+10, angle.time_support.start[0]+20),
#     'rem':nap.IntervalSet(rem_ep.start[0], rem_ep.start[0]+2),
#     'sws':nap.IntervalSet(sws_ep.start[0], sws_ep.start[0]+2)
#     }

filepath = os.path.join(os.path.expanduser("~"), f'Dropbox/LMNphysio/data/DATA_FIG_LMN_ADN_{basename}.pickle')
exdata = cPickle.load(open(filepath, 'rb'))
exs = {"wak": exdata["ex_wak"], "rem": exdata["ex_rem"], "sws": exdata["ex_sws"]}

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
    X_exs[name] = {}

    for loc in groups.keys():

        X = groups[loc].count(bin_sizes[name], epochs)
        # X = np.sqrt(groups[loc].count(bin_sizes[name], epochs))
        # X = np.log(1+groups[loc].count(bin_sizes[name], epochs))
        X = X / X.max(0)
        X = X.smooth(bin_sizes[name]*1, norm=False)
        # X = X - X.mean(0)
        # X = X / X.std(0)
        # X = X/X.max(0)
        Xex = X.restrict(exs[name])
        # Dropping empty bins for sleep
        if name == 'sws' or name == 'rem':
            # threshold = np.percentile(X.sum(1), 90)
            # X = X[X.sum(1)>threshold]
            # print(np.percentile(X.mean(1), 50))
            thr = np.percentile(X.mean(1), 40)
            X = X[X.mean(1) > thr]
            # Xex = Xex[Xex.mean(1) > thr]

        X = X - X.mean(0)
        X = X / X.std(0)

        X_[name][loc] = X
        X_exs[name][loc] = Xex

# %%
# --------------------------
# Dimensionality reduction through autoencoder model
# --------------------------

class NeuralAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=2):
        super().__init__()
        
        # Encoder: high-dim -> 2D
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: 2D -> high-dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction
    
    def encode(self, x):
        """Just get the 2D projection"""
        return self.encoder(x)



class PolarNeuralAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        u, v = z[:, 0], z[:, 1]

        r = torch.sqrt(u**2 + v**2 + 1e-8)
        theta = torch.atan2(v, u)

        x_hat = self.decoder(z)
        return z, x_hat, r, theta


proj = {"wak": {}, "sws": {}, "rem": {}}
proj_ex = {"wak": {}, "sws": {}, "rem": {}}

losses = []

for loc in ["adn", "lmn"]:
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_dim = X_['wak'][loc].shape[1]

    # model = PolarNeuralAutoencoder(input_dim)

    model = NeuralAutoencoder(input_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    criterion = nn.MSELoss()
    #
    # # Loss weights
    # alpha = 1.0     # ring constraint
    # beta = 0.1      # temporal smoothness
    # r0 = 1.0        # target radius
    #
    X = torch.FloatTensor(X_['wak'][loc].values)

    n_epochs = 3000

    for epoch in range(n_epochs):
        # optimizer.zero_grad()

        # z, x_hat, r, theta = model(X)
        embedding, reconstruction = model(X)

        loss = criterion(reconstruction, X)

        # # Reconstruction loss
        # recon_loss = ((X - x_hat)**2).mean()
        #
        # # Ring constraint
        # ring_loss = ((r - r0)**2).mean()
        #
        # # Temporal smoothness
        # smooth_loss = ((z[1:] - z[:-1])**2).mean()
        #
        # # Total loss
        # loss = recon_loss + alpha * ring_loss + beta * smooth_loss
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.6f}")
        

    # 2d projections
    model.eval()
    with torch.no_grad():
        for name in ['wak', 'sws', 'rem']:
            data_tensor = torch.FloatTensor(X_[name][loc].values)
            embedding, _ = model(data_tensor)
            proj[name][loc] = embedding.numpy()
            ex_tensor = torch.FloatTensor(X_exs[name][loc].values)
            embedding_ex, _ = model(ex_tensor)
            proj_ex[name][loc] = embedding_ex.numpy()



colors = getRGB(angle, newwake_ep, bin_size=bin_sizes['wak'])


# # Save data
# datatosave = {
#     'proj': proj,
#     'proj_ex': proj_ex,
#     'colors': colors
# }
# dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"
# cPickle.dump(datatosave, open(os.path.join(dropbox_path, "projections_LMN_ADN_v4.pickle"), "wb"))

#%%
fig = plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")


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

for j, name in enumerate(['wak', 'rem', 'sws']):
    for i, loc in enumerate(['adn', 'lmn']):
        ax = fig.add_subplot(gs[i, j])
        Y = proj[name][loc]

        if name == 'wak':
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
plt.savefig(os.path.expanduser("~/Dropbox/LMNphysio/data/projections_scatter_KPCA_v4.png"), dpi=300)


#%%
# --------------------------
# Histogram (2D) of projections
# --------------------------
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(2, 3, wspace=0.3, hspace=0.3)

for j, name in enumerate(['wak', 'rem', 'sws']):
    for i, loc in enumerate(['adn', 'lmn']):
        ax = fig.add_subplot(gs[i, j], aspect='equal')
        Y = proj[name][loc]
        x = Y[:, 0]
        y = Y[:, 1]
        # bins = [np.linspace(-1.2, 1.2, 50), np.linspace(-1.2, 1.2, 50)]
        H, xedges, yedges = np.histogram2d(x, y, bins=50, density=True)
        # H = np.log(1 + H)
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
plt.savefig(os.path.expanduser("~/Dropbox/LMNphysio/data/projections_hist_KPCA_v4.png"), dpi=300)
plt.show()

#%%
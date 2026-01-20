# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-17 10:02:32
"""
N LMN -> N ADN 
Non linearity + CAN Current + inhibition in ADN + PSB Feedback
Population coherence

"""

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import jit, njit
import os
import pandas as pd
from model import Model


def smoothAngularTuningCurves(tuning_curves, window=20, deviation=3.0):
    new_tuning_curves = tuning_curves.copy(deep=True)
    tmp = []
    for i in tuning_curves.unit:
        tcurves = tuning_curves.sel(unit=i)
        index = tcurves.coords[tcurves.dims[0]].values
        offset = np.mean(np.diff(index))
        padded = pd.Series(index=np.hstack((index - (2 * np.pi) - offset,
                                            index,
                                            index + (2 * np.pi) + offset)),
                           data=np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=window, win_type='gaussian', center=True, min_periods=1).mean(std=deviation)
        tmp.append(smoothed.loc[index])

    new_tuning_curves.values = np.array(tmp)

    return new_tuning_curves


# 5
# np.random.seed(5)
np.random.seed(42)

N_t = 10000

m_wake = Model(D_lmn=0.0, noise_lmn_ = 1.0, N_t=N_t)  # Wake
m_wake.run()

m_sws = Model(I_lmn=0.0, N_t=N_t)  # Sleep
m_sws.run()

m_opto = Model(I_lmn=0.0, w_psb_lmn_=0.0, N_t=N_t)  # Opto
m_opto.run()

popcoh = {}
for k in ['adn', 'lmn']:
    popcoh[k] = {}
    for n, m in zip(['wak', 'sws', 'opto'], [m_wake, m_sws, m_opto]):
        r = getattr(m, f"r_{k}")
        r_smooth = gaussian_filter1d(r, sigma=1, axis=0)
        # r_smooth = r
        tmp = np.corrcoef(r_smooth.T)
        popcoh[k][n] = tmp[np.triu_indices(tmp.shape[0], 1)]
    popcoh[k] = pd.DataFrame.from_dict(popcoh[k])

og_adn = [0.91, 0.39]
og_lmn = [0.74, 0.08]
mod_p = {"adn": [], "lmn": []}
figure(figsize=(12, 5))
gs = GridSpec(2, 2)
for i, st in enumerate(popcoh.keys()):
    gs2 = GridSpecFromSubplotSpec(1, 2, gs[0, i])
    for j, e in enumerate(['sws', 'opto']):
        subplot(gs2[0, j])
        gca().set_aspect("equal")
        plot(popcoh[st]['wak'], popcoh[st][e], 'o')
        r, p = pearsonr(popcoh[st]['wak'], popcoh[st][e])
        m, b = np.polyfit(popcoh[st]['wak'], popcoh[st][e], 1)
        x = np.linspace(popcoh[st]['wak'].min(), popcoh[st]['wak'].max(), 5)
        plot(x, x * m + b)

        xlim(-1, 1)
        ylim(-1, 1)
        title(st + " " + e + f" r={np.round(r, 2)}")

        mod_p[st].append(r)

ax = subplot(gs[1, 0])
ax.plot([0, 1], og_adn, '+--', label='ADN', color='C0')
ax.plot([3, 4], og_lmn, '+--', label='LMN', color='C1')
ax.plot([0, 1], mod_p['adn'], 'o-', label='ADN Model', color='C0')
ax.plot([3, 4], mod_p['lmn'], 'o-', label='LMN Model', color='C1')
ax.set_xticks([0, 1, 3, 4])
ax.set_ylabel('Population Coherence Correlation')
ax.legend()

# durations = [2000, 1000, 200]  # For m_wake, m_sws, m_opto
# durations = [3000, 3000, 200]
durations = [
    slice(100, 3000),
    slice(100, 3000),
    slice(100, 400),
]

figure(figsize=(14, 8))
gs = GridSpec(1, 3, wspace=0.4)

for i, (m, duration) in enumerate(zip([m_wake, m_sws, m_opto], durations)):
    n_rows = 6
    inner_gs = gs[i].subgridspec(n_rows, 1, hspace=0.4)

    ax0 = subplot(inner_gs[0])
    # ax0.plot(m.r_lmn[:duration], '-')
    tmp = m.r_lmn[duration].T
    # tmp = gaussian_filter1d(tmp, sigma=1, axis=0)
    tmp = gaussian_filter(tmp, sigma=3)
    ax0.pcolormesh(tmp, cmap='jet', shading='auto')
    ax0.set_ylabel("r_lmn")
    if i == 0:
        ax0.set_title("Wake")
    elif i == 1:
        ax0.set_title("SWS")
    else:
        ax0.set_title("Opto")

    ax1 = subplot(inner_gs[1], sharex=ax0)
    ax1.plot(m.x_adn[duration], '-')
    ax1.axhline(m.thr_adn, linestyle='--', color='k')
    ax1.set_ylabel("x_adn")

    ax2 = subplot(inner_gs[2], sharex=ax0)
    ax2.plot(m.r_adn[duration], '-')
    ax2.set_ylabel("r_adn")

    ax3 = subplot(inner_gs[3], sharex=ax0)
    tmp = m.r_adn[duration].T
    tmp = gaussian_filter(tmp, sigma=3)
    pcm = ax3.pcolormesh(tmp, cmap='jet', shading='auto')#, vmin=0, vmax=1)
    ax3.set_ylabel("r_adn")
    # colorbar(pcm, ax=ax3)

    ax4 = subplot(inner_gs[4], sharex=ax0)
    ax4.plot(m.r_trn[duration], '-', color='red')
    ax4.plot(m.x_trn[duration], '--', color='gray')
    ax4.set_ylabel("r_trn")

    ax5 = subplot(inner_gs[5], sharex=ax0)
    ax5.plot(m.I_ext[duration])
    ax5.set_ylabel("I ADN")

# ============================================================
# Pynapple decoding and HSV-colored ring visualization
# ============================================================
import pynapple as nap
import matplotlib.cm as cm
import cmocean

# Create time array (assuming dt = tau = 0.1)
dt = m_wake.tau
time = np.arange(N_t) * dt

# Define the angle based on the input bump position (rotating every 50 timesteps)
# The bump rotates through N_lmn positions
angle = np.linspace(0, 2 * np.pi, m_wake.N_lmn)[np.argmax(m_wake.inp_lmn, 1)]

# Create pynapple objects for wake condition
angle_tsd = nap.Tsd(t=time, d=angle)
angle_tsd = np.unwrap(angle_tsd).smooth(std=dt * 3) % (2 * np.pi)

wake_epoch = nap.IntervalSet(start=0, end=time[-1])

# Convert r_adn to TsdFrame for each condition
r_wake = nap.TsdFrame(t=time, d=m_wake.r_adn)
r_sws = nap.TsdFrame(t=time, d=m_sws.r_adn)
r_opto = nap.TsdFrame(t=time, d=m_opto.r_adn)

# Compute tuning curves during wake
n_bins = 120
tc = nap.compute_tuning_curves(r_wake, angle_tsd, bins=n_bins, range=(0, 2 * np.pi), epochs=wake_epoch,
                               feature_names=["angle"])  #, return_pandas=True)
tc2 = smoothAngularTuningCurves(tc, window=30, deviation=3)

# Decode angle for SWS and Opto conditions using tuning curves from wake
sws_epoch = nap.IntervalSet(start=0, end=time[-1])
opto_epoch = nap.IntervalSet(start=0, end=time[-1])

decoded_sws, _ = nap.decode_template(tc, r_sws, sws_epoch, bin_size=dt)
decoded_opto, _ = nap.decode_template(tc, r_opto, opto_epoch, bin_size=dt)

# HSV-colored ring visualization
figure(figsize=(15, 5))
gs_hsv = GridSpec(2, 3, hspace=0.3, wspace=0.3)

# For wake, use actual angle; for sws/opto use decoded angle
angles_dict = {
    'Wake': angle_tsd.values,
    'SWS': decoded_sws.values,
    'Opto': decoded_opto.values
}

# Idx for opto condition to match length after decoding
r_data = m_opto.r_adn
tmp = r_data.mean(1)
mask_opto = tmp >= np.percentile(tmp, 50)
angles_dict['Opto'] = angles_dict['Opto'][mask_opto]

# Idx for SWS condition to match length after decoding
r_data = m_sws.r_adn
tmp = r_data.mean(1)
mask_sws = tmp >= np.percentile(tmp, 50)
angles_dict['SWS'] = angles_dict['SWS'][mask_sws]

# embeddings = {}
# for row, region in enumerate(['lmn', 'adn']):
#     r_data = getattr(m, f"r_{region}")
#     isomap = KernelPCA(n_components=2, kernel='cosine')
#     r_data_smooth = r_data
#     r_data_smooth = StandardScaler().fit_transform(r_data_smooth)
#     isomap.fit(r_data_smooth[1000:])
#
#     embeddings[region] = isomap

embeddings = {}
for row, region in enumerate(['lmn', 'adn']):
    embeddings[region] = {}
    for col, (name, m) in enumerate(zip(['Wake', 'SWS', 'Opto'], [m_wake, m_sws, m_opto])):
        r_data = getattr(m, f"r_{region}")
        # r_data = np.sqrt(r_data)  # variance stabilization

        isomap = KernelPCA(n_components=2, kernel='cosine')
        # r_data_smooth = gaussian_filter1d(r_data, sigma=2, axis=0)
        r_data_smooth = r_data
        r_data_smooth = StandardScaler().fit_transform(r_data_smooth)

        if name == 'Opto':
            r_data_smooth = r_data_smooth[mask_opto]
        elif name == 'SWS':
            r_data_smooth = r_data_smooth[mask_sws]

        embedding = isomap.fit_transform(r_data_smooth[1000:])
        # embedding = embeddings[region].transform(r_data_smooth[1000:])

        embeddings[region][name.lower()] = embedding

        # Normalize angle to [0, 1] for twilight colormap
        ang = angles_dict[name][1000:]
        ang_normalized = (ang % (2 * np.pi)) / (2 * np.pi)
        rgb_colors = cmocean.cm.phase(ang_normalized)
        rgb_colors = cm.twilight_shifted(ang_normalized)

        # select random points
        idx = np.random.choice(np.arange(len(embedding)), size=4000, replace=False)

        ax = subplot(gs_hsv[row, col])
        # Shuffle the order
        np.random.shuffle(idx)
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c=rgb_colors[idx], s=5, alpha=1)

        if row == 0:
            ax.set_title(name)
        ax.set_xlabel("Isomap 1")
        if col == 0:
            ax.set_ylabel(f"r_{region}\nIsomap 2")

suptitle("Isomap projections colored by decoded angle (twilight)")
tight_layout()

show()

datatosave = {
    # Parameters
    "parameters": m_wake.get_parameters(),
    # Firing rates
    "lmn": {e: m.r_lmn for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    "adn": {e: m.r_adn for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    "trn": {e: m.r_trn for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    # Membrane potentials
    "x_adn": {e: m.x_adn for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    "x_trn": {e: m.x_trn for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    # External input to ADN
    "I_ext": {e: m.I_ext for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    # Threshold
    "thr_adn": m_wake.thr_adn,
    # Population coherence
    "popcoh": popcoh,
    # Time parameters
    "N_t": N_t,
    "dt": dt,
    "time": time,
    "durations": durations,
    # Angles for coloring
    "angle": angle,
    "angle_tsd": angle_tsd.values,
    "decoded_sws": decoded_sws.values,
    "decoded_opto": decoded_opto.values,
    "mask_opto": mask_opto,
    "mask_sws": mask_sws,
    # Tuning curves
    "tc": tc,
    "tc2": tc2,
    # Embeddings from last figure
    "embeddings": embeddings,
}
import _pickle as cPickle

# filepath = os.path.join(os.path.expanduser("~") + "/Dropbox/LMNphysio/model/model_rings.pickle")
# cPickle.dump(datatosave, open(filepath, 'wb'))

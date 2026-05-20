# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2026-04-14
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from functions import *

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter


###############################################################################################
# GENERAL infos
###############################################################################################
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

datasets = np.unique(np.hstack([
    np.genfromtxt(os.path.join(data_directory, 'datasets_LMN_ADN.list'),
                  delimiter='\n', dtype=str, comments='#'),
]))

###############################################################################################
# Accumulators  —  one entry per neuron
###############################################################################################
allpisi = {
    'adn': {'up': [], 'down': []},
    'lmn': {'up': [], 'down': []},
}
tcurves_up  = {'adn': [], 'lmn': []}
tcurves_down = {'adn': [], 'lmn': []}

bins = np.geomspace(0.002, 100.0, 200)
nb_bin_hd = 32
bin_size_sws = 0.01

for s in datasets:
    print(s)
    ###############################################################################################
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

    if not os.path.exists(filepath):
        continue

    nwb = nap.load_file(filepath)

    spikes = nwb['units']
    spikes = spikes.getby_threshold("rate", 1)

    position = []
    columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    for k in columns:
        position.append(nwb[k].values)
    position = np.array(position)
    position = np.transpose(position)
    position = nap.TsdFrame(
        t=nwb['x'].t,
        d=position,
        columns=columns,
        time_support=nwb['position_time_support'])

    epochs = nwb['epochs']
    sws_ep = nwb['sws']
    rem_ep = nwb['rem']

    nwb.close()

    # UP / DOWN epochs from neuroscope annotations
    try:
        up_ep   = read_neuroscope_intervals(path, basename, 'up')
        down_ep = read_neuroscope_intervals(path, basename, 'down')
    except Exception:
        print(f"  Skipping {s}: no UP/DOWN intervals found")
        continue

    # Keep only ADN + LMN neurons
    spikes = spikes[(spikes.location == "lmn") | (spikes.location == "adn")]

    if not len(spikes):
        continue

    ###############################################################################################
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(
        spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
        ep=position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

    SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'],
                                    position.time_support.loc[[0]],
                                    minmax=(0, 2 * np.pi))
    spikes.set_info(SI)
    spikes = spikes[spikes.SI > 0.1]

    # CHECKING HALF EPOCHS
    wake2_ep = splitWake(position.time_support.loc[[0]])
    tokeep2 = []
    for i in range(2):
        tcurves_half = nap.compute_1d_tuning_curves(
            spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
            ep=wake2_ep[i])
        tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)
        tokeep, _ = findHDCells(tcurves_half)
        tokeep2.append(tokeep)
    tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

    print(f"  {s}  kept: {len(tokeep)}")

    if len(tokeep) < 5:
        continue

    spikes  = spikes[tokeep]
    tcurves = tuning_curves[tokeep]

    adn = spikes.index[spikes.location == "adn"]
    lmn = spikes.index[spikes.location == "lmn"]

    if not len(adn) and not len(lmn):
        continue

    try:
        velocity = computeLinearVelocity(position[['x', 'z']],
                                         position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
    except Exception:
        velocity = computeAngularVelocity(position['ry'],
                                          position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)

    ###############################################################################################
    # DECODE HEAD DIRECTION DURING SWS
    ###############################################################################################
    # count = (spikes
    #          .count(bin_size_sws)
    #          .smooth(bin_size_sws * 2, size_factor=20, norm=False)
    #          .restrict(sws_ep))
    # sws_angle, _ = nap.decode_1d(tcurves, count, sws_ep, bin_size_sws)

    xcurves  = nap.compute_tuning_curves(spikes, position['ry'], bins=120, range=(0, 2 * np.pi), feature_names=['angle'], epochs=newwake_ep)
    sws_angle, _ = nap.decode_bayes(xcurves, spikes, sws_ep, bin_size_sws, sliding_window_size=3)


    # Interpolate to spike timestamps
    ref_spk = spikes[tokeep[0]]
    sws_angle2 = np.unwrap(sws_angle).interpolate(ref_spk.count(0.002, sws_ep))
    sws_angle2 = sws_angle2 % (2 * np.pi)

    # # Active-period mask (require ≥1.5 spk/bin summed across population)
    # count_thresh = (spikes
    #                 .count(bin_size_sws)
    #                 .smooth(bin_size_sws, size_factor=20, norm=False)
    #                 .restrict(sws_ep))
    # sumcount = count_thresh.sum(1)
    # active_sws_ep = sumcount.threshold(1.5).time_support.drop_short_intervals(0.05)
    #
    # # Intersect UP / DOWN with active SWS windows
    # active_up_ep   = up_ep.intersect(active_sws_ep)
    # active_down_ep = down_ep.intersect(active_sws_ep)
    #
    # if active_up_ep.tot_length('s') < 5 or active_down_ep.tot_length('s') < 5:
    #     print(f"  Skipping {s}: not enough UP/DOWN activity")
    #     continue

    ###############################################################################################
    # COMPUTE pISI FOR EACH GROUP × STATE
    ###############################################################################################
    for name, group in [('adn', adn), ('lmn', lmn)]:
        if not len(group):
            continue

        # --- UP ---
        pisi_up, xbins, ybins = compute_ISI_HD(
            spikes[group], sws_angle2, up_ep,
            bins=bins, nb_bin_hd=nb_bin_hd)
        pisi_up = np.array([pisi_up[n].values for n in pisi_up.keys()])
        allpisi[name]['up'].append(pisi_up)

        tc_up = nap.compute_1d_tuning_curves(
            spikes[group], sws_angle2, 120,
            minmax=(0, 2 * np.pi), ep=up_ep)
        tc_up = smoothAngularTuningCurves(tc_up, window=20, deviation=2.0)
        tcurves_up[name].append(tc_up)

        # --- DOWN ---
        pisi_down, xbins, ybins = compute_ISI_HD(
            spikes[group], sws_angle2, down_ep,
            bins=bins, nb_bin_hd=nb_bin_hd)
        pisi_down = np.array([pisi_down[n].values for n in pisi_down.keys()])
        allpisi[name]['down'].append(pisi_down)

        tc_down = nap.compute_1d_tuning_curves(
            spikes[group], sws_angle2, 120,
            minmax=(0, 2 * np.pi), ep=down_ep)
        tc_down = smoothAngularTuningCurves(tc_down, window=20, deviation=2.0)
        tcurves_down[name].append(tc_down)


###############################################################################################
# STACK
###############################################################################################
for name in ['adn', 'lmn']:
    if allpisi[name]['up']:
        allpisi[name]['up']   = np.vstack(allpisi[name]['up'])
        allpisi[name]['down'] = np.vstack(allpisi[name]['down'])
        tcurves_up[name]   = pd.concat(tcurves_up[name], axis=1)
        tcurves_down[name] = pd.concat(tcurves_down[name], axis=1)

        tcurves_up[name].columns   = np.arange(tcurves_up[name].shape[1])
        tcurves_up[name]   = centerTuningCurves_with_peak(tcurves_up[name])
        tcurves_down[name].columns = np.arange(tcurves_down[name].shape[1])
        tcurves_down[name] = centerTuningCurves_with_peak(tcurves_down[name])


###############################################################################################
# FIGURE
###############################################################################################
extents = [xbins[0], xbins[-1], ybins[-1], ybins[0]]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
xt = [np.argmin(np.abs(bins - x)) for x in [10**-2, 1]]
# for row, name in enumerate(['adn', 'lmn']):
for row, name in enumerate(['adn']):
    if not isinstance(allpisi[name]['up'], np.ndarray):
        continue

    for col, state in enumerate(['up', 'down']):
        ax = axes[col]
        tmp = allpisi[name][state].mean(0)
        tmp = tmp / (tmp.sum(0) + 1e-12)
        tmp = gaussian_filter(tmp, sigma=1)
        ax.imshow(tmp, aspect='auto', cmap='jet', origin= 'lower')
        ax.set_title(f"{name.upper()} — {state}")
        ax.set_xlabel("HD bin")
        ax.set_ylabel("ISI (s)")
        ax.set_xticks([0, tmp.shape[1] // 2, tmp.shape[1] - 1], ['-180', '0', '180'])
        ax.set_yticks(xt, ['0.01', '1'])

    ax = axes[2]
    ax.semilogx(ybins[0:-1], allpisi[name]['up'].mean(0).sum(1),
                label='up')
    ax.semilogx(ybins[0:-1], allpisi[name]['down'].mean(0).sum(1),
                label='down')
    ax.legend()
    ax.set_title(f"{name.upper()} — marginal ISI")
    ax.set_xlabel("ISI (s)")
    ax.set_ylabel("density")

plt.tight_layout()
plt.savefig(os.path.expanduser(
    "~/Dropbox/LMNphysio/data/pisi_up_down.png"))
# plt.show()


###############################################################################################
# SAVE
###############################################################################################
# datatosave = {
#     'pisi': allpisi,
#     'bins': bins,
#     'xbins': xbins,
#     'ybins': ybins,
#     'tcurves_up':   tcurves_up,
#     'tcurves_down': tcurves_down,
# }
#
# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# cPickle.dump(datatosave,
#              open(os.path.join(dropbox_path, 'PISI_UP_DOWN.pickle'), 'wb'))
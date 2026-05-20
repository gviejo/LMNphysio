"""
This script should :
1. Load LMN-ADN dataset
2. Compute tuning curves for ADN and LMN cells during wak_ep using nap.compute_tuning_curves from pynapple
2. Decode head direction from ADN cells during sws_ep using nap.decode_bayes from pynapple
3. In bins of 10 ms, generate random poisson spike trains for each LMN cell based on their tuning curves.
4. Compute tuning curves for the generated LMN spike trains during sws_ep using nap.compute_tuning_curves from pynapple
5. Compare the tuning curves of the generated LMN spike trains to the tuning curves of the LMN cells during sws_ep

"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from functions import *

import _pickle as cPickle
from matplotlib.gridspec import GridSpec


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

datasets = np.unique(
    np.genfromtxt(os.path.join(data_directory, 'datasets_LMN_ADN.list'),
                  delimiter='\n', dtype=str, comments='#')
)

bin_size_sws = 0.01  # 10 ms

# Accumulators  (list of 2-D arrays: shape (n_angle_bins, n_neurons_in_session))
all_center_tc_real = []   # UP state, real spikes
all_center_tc_fake = []   # UP state, Poisson prediction
all_center_tc_wak_real = []   # wake, real spikes
all_center_tc_wak_fake = []   # wake, Poisson prediction

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

    epochs  = nwb['epochs']
    sws_ep  = nwb['sws']

    nwb.close()

    sws_ep = read_neuroscope_intervals(path, basename, 'up')

    spikes = spikes[(spikes.location == "lmn") | (spikes.location == "adn")]
    if not len(spikes):
        continue

    ###############################################################################################
    # STEP 2 — TUNING CURVES DURING WAKE (nap.compute_tuning_curves)
    ###############################################################################################
    try:
        velocity = computeLinearVelocity(position[['x', 'z']],
                                         position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
    except Exception:
        velocity = computeAngularVelocity(position['ry'],
                                          position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)

    tuning_curves = nap.compute_tuning_curves(
        spikes, position['ry'],
        bins=120, range=(0, 2 * np.pi),
        feature_names=['angle'],
        epochs=newwake_ep)

    # Also compute smooth 1-d version for cell selection and plotting
    tuning_curves_1d = nap.compute_1d_tuning_curves(
        spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
        ep=position.time_support.loc[[0]])
    tuning_curves_1d = smoothAngularTuningCurves(tuning_curves_1d, 20, 4)

    SI = nap.compute_1d_mutual_info(tuning_curves_1d, position['ry'],
                                    position.time_support.loc[[0]],
                                    minmax=(0, 2 * np.pi))
    spikes.set_info(SI)
    spikes = spikes[spikes.SI > 0.1]

    # Half-epoch stability check
    wake2_ep = splitWake(position.time_support.loc[[0]])
    tokeep2 = []
    for i in range(2):
        tc_half = nap.compute_1d_tuning_curves(
            spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
            ep=wake2_ep[i])
        tc_half = smoothAngularTuningCurves(tc_half, 20, 4)
        tokeep_i, _ = findHDCells(tc_half)
        tokeep2.append(tokeep_i)
    tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

    spikes = spikes[tokeep]

    adn = spikes.index[spikes.location == "adn"]
    lmn = spikes.index[spikes.location == "lmn"]

    if not len(adn) or not len(lmn):
        continue

    print(f"  adn={len(adn)}  lmn={len(lmn)}")

    tuning_curves_1d = nap.compute_1d_tuning_curves(
        spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
        ep=newwake_ep)
    tuning_curves_1d = smoothAngularTuningCurves(tuning_curves_1d, 20, 4)

    ###############################################################################################
    # STEP 3 — DECODE HD FROM ADN DURING SWS (nap.decode_bayes)
    ###############################################################################################
    # Refresh tuning curves for the kept neurons only
    tc_adn = nap.compute_tuning_curves(
        spikes[adn], position['ry'],
        bins=120, range=(0, 2 * np.pi),
        feature_names=['angle'],
        epochs=newwake_ep)

    sws_angle, _ = nap.decode_bayes(
        tc_adn, spikes[adn], sws_ep,
        bin_size=bin_size_sws,
        sliding_window_size=3)

    # ybins used to bin angles for Poisson generation
    angle_bins = tuning_curves_1d.index.values          # bin centres from compute_tuning_curves
    # (shape: nb_bins,)

    ###############################################################################################
    # STEP 4 — GENERATE RANDOM POISSON SPIKE TRAINS FOR LMN CELLS DURING SWS
    ###############################################################################################
    tc_lmn_wak = tuning_curves_1d[lmn]  # shape: (120, n_lmn)

    digit_angle = np.digitize(sws_angle, tuning_curves.bin_edges[0])

    lmn_rates_sws = []

    for n in lmn:
        tc_n = tc_lmn_wak[n]  # pd.Series, index = angle bins (rad)

        rates_at_angle = tc_n.values[digit_angle.values - 1]  # shape: (nb_bins,) — expected rate for each decoded angle bin

        lmn_rates_sws.append(rates_at_angle)

    lmn_rates_sws = np.array(lmn_rates_sws)  # shape: (n_lmn, nb_bins)
    lmn_rates_sws = nap.TsdFrame(t=sws_angle.index.values, d=lmn_rates_sws.T, columns=lmn, time_support=sws_ep)

    ###############################################################################################
    # GENERATE RANDOM POISSON SPIKE TRAINS FOR LMN CELLS DURING WAKE
    ###############################################################################################
    digit_angle = np.digitize(position['ry'], tuning_curves.bin_edges[0])

    lmn_rates_wak = []

    for n in lmn:
        tc_n = tc_lmn_wak[n]  # pd.Series, index = angle bins (rad)

        rates_at_angle = tc_n.values[digit_angle.values - 1]  # shape: (nb_bins,) — expected rate for each decoded angle bin

        lmn_rates_wak.append(rates_at_angle)

    lmn_rates_wak = np.array(lmn_rates_wak)  # shape: (n_lmn, nb_time_bins)
    lmn_rates_wak = nap.TsdFrame(t=position['ry'].index.values, d=lmn_rates_wak.T, columns=lmn,
                                  time_support=position.time_support)

    ###############################################################################################
    # STEP 5 — COMPUTE TUNING CURVES FOR GENERATED SPIKES DURING UP STATES
    ###############################################################################################
    tc_lmn_sws_real = nap.compute_tuning_curves(
        spikes[lmn], sws_angle,
        bins=120, range=(0, 2 * np.pi),
        feature_names=['angle'],
        epochs=sws_ep, return_pandas=True)

    tc_lmn_sws_fake = nap.compute_tuning_curves(
        lmn_rates_sws, sws_angle,
        bins=120, range=(0, 2 * np.pi),
        feature_names=['angle'],
        epochs=sws_ep, return_pandas=True)

    # SAME FOR WAKE
    tc_lmn_wak_real = nap.compute_tuning_curves(
        spikes[lmn], position['ry'],
        bins=120, range=(0, 2 * np.pi),
        feature_names=['angle'],
        epochs=newwake_ep, return_pandas=True)

    tc_lmn_wak_fake = nap.compute_tuning_curves(
        lmn_rates_wak, position['ry'],
        bins=120, range=(0, 2 * np.pi),
        feature_names=['angle'],
        epochs=newwake_ep, return_pandas=True)

    ###############################################################################################
    # STEP 6 : Center tuning curves
    ###############################################################################################
    center_tc_real = centerTuningCurves_with_peak(tc_lmn_sws_real)
    center_tc_fake = centerTuningCurves_with_peak(tc_lmn_sws_fake)
    center_tc_wak_real = centerTuningCurves_with_peak(tc_lmn_wak_real)
    center_tc_wak_fake = centerTuningCurves_with_peak(tc_lmn_wak_fake)

    all_center_tc_real.append(center_tc_real.values)
    all_center_tc_fake.append(center_tc_fake.values)
    all_center_tc_wak_real.append(center_tc_wak_real.values)
    all_center_tc_wak_fake.append(center_tc_wak_fake.values)


###############################################################################################
# STACK ACROSS SESSIONS  (n_angle_bins, N_neurons_total)
###############################################################################################
all_center_tc_real     = np.hstack(all_center_tc_real)
all_center_tc_fake     = np.hstack(all_center_tc_fake)
all_center_tc_wak_real = np.hstack(all_center_tc_wak_real)
all_center_tc_wak_fake = np.hstack(all_center_tc_wak_fake)


###############################################################################################
# FIGURE  — 2 rows (UP state | wake) × 2 columns (real | Poisson) + mean overlay
###############################################################################################
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

n_bins = all_center_tc_real.shape[0]
x = np.linspace(-180, 180, n_bins)

for row, (label, real_arr, fake_arr) in enumerate([
        ('UP state', all_center_tc_real, all_center_tc_fake),
        ('Wake',     all_center_tc_wak_real, all_center_tc_wak_fake),
]):
    m_real = np.nanmean(real_arr, 1)
    s_real = np.nanstd(real_arr, 1) / np.sqrt(np.sum(np.isfinite(real_arr), 1))
    m_fake = np.nanmean(fake_arr, 1)
    s_fake = np.nanstd(fake_arr, 1) / np.sqrt(np.sum(np.isfinite(fake_arr), 1))

    # Column 0 : mean curves
    ax = axes[row, 0]
    ax.plot(x, m_real, color='k', lw=2, label='Real')
    ax.fill_between(x, m_real - s_real, m_real + s_real, color='k', alpha=0.2)
    ax.plot(x, m_fake, color='r', lw=2, label='Poisson')
    ax.fill_between(x, m_fake - s_fake, m_fake + s_fake, color='r', alpha=0.2)
    ax.set_title(f'{label} — mean TC  (n={real_arr.shape[1]})')
    ax.set_xlabel('Angle from peak (°)')
    ax.set_ylabel('Norm. rate')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    # Column 1 : difference (real − Poisson)
    ax = axes[row, 1]
    diff = m_real - m_fake
    ax.plot(x, diff, color='steelblue', lw=2)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.fill_between(x, diff, 0, where=diff > 0, color='steelblue', alpha=0.3)
    ax.fill_between(x, diff, 0, where=diff < 0, color='tomato',    alpha=0.3)
    ax.set_title(f'{label} — Real − Poisson')
    ax.set_xlabel('Angle from peak (°)')
    ax.set_ylabel('Δ norm. rate')
    ax.spines[['top', 'right']].set_visible(False)

plt.savefig(os.path.expanduser(
    "~/Dropbox/LMNphysio/data/random_tc_comparison.png"),
    dpi=150, bbox_inches='tight')
plt.show()

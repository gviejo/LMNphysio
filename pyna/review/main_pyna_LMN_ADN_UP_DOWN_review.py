# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-05-13 17:23:05

# %%
import numpy as np
import pandas as pd
import pynapple as nap
# import nwbmatic as ntm
from pylab import *
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.stats import zscore, wilcoxon, mannwhitneyu



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

datasets = np.hstack([
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)



frates = {'adn': [], 'lmn': []}


for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)

    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

    if os.path.exists(filepath):

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
        wake_ep = epochs[epochs.tags == "wake"]
        
        sws_ep = nwb['sws']
        rem_ep = nwb['rem']

        nwb.close()
        
        up_ep = read_neuroscope_intervals(path, basename, 'up')
        down_ep = read_neuroscope_intervals(path, basename, 'down')

        spikes = spikes[(spikes.location == "lmn") | (spikes.location    == "adn")]


        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(
            spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
            ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        
        SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
        spikes.set_info(SI)

        spikes = spikes[spikes.SI>0.1]

        # CHECKING HALF EPOCHS
        wake2_ep = splitWake(position.time_support.loc[[0]])    
        tokeep2 = []
        stats2 = []
        tcurves2 = []   
        for i in range(2):
            tcurves_half = nap.compute_1d_tuning_curves(
                spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                ep = wake2_ep[i]
                )
            tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

            tokeep, stat = findHDCells(tcurves_half)
            tokeep2.append(tokeep)
            stats2.append(stat)
            tcurves2.append(tcurves_half)       
        tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
    
        if len(tokeep) > 4:

            spikes = spikes[tokeep]
            # groups = spikes._metadata.loc[tokeep].groupby("location").groups
            tcurves         = tuning_curves[tokeep]

            try:
                velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
            except:
                velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)


            #########################
            # FIRING RATE MODULATION
            #########################
            adn = spikes.index[spikes.location == 'adn']
            lmn = spikes.index[spikes.location == 'lmn']

            if len(adn) > 0 and len(lmn) > 0:

                for name, group in zip(['adn', 'lmn'], [adn, lmn]):

                    up_rate = spikes[group].restrict(up_ep).rate
                    down_rate = spikes[group].restrict(down_ep).rate
                    delta = (down_rate - up_rate)/up_rate

                    tmp = pd.DataFrame({
                        "up": up_rate,
                        "down": down_rate,
                        "delta": delta,
                        })
                    tmp.index = [s.split("/")[-1] +"_"+str(i) for i in tmp.index]

                    frates[name].append(tmp)


for name in frates.keys():
    frates[name] = pd.concat(frates[name], axis = 0)


# %%
# --------------------------
# Figure
# --------------------------
col_adn  = '#5B9BD5'   # blue  — ADN
col_lmn  = '#ED7D31'   # orange — LMN

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.subplots_adjust(wspace=0.45)

# --- Panels 1 & 2 : paired up/down rates per group ---
for ax, name, col in zip(axes[:2], ['adn', 'lmn'], [col_adn, col_lmn]):
    up_vals   = frates[name]['up'].values
    down_vals = frates[name]['down'].values

    for u, d in zip(up_vals, down_vals):
        ax.plot([0, 1], [u, d], '-', color='grey', alpha=0.3, lw=0.8)

    # mean line
    ax.plot([0, 1], [up_vals.mean(), down_vals.mean()], '-', color='k', lw=2, zorder=4)

    # mean ± SEM
    for x, vals in [(0, up_vals), (1, down_vals)]:
        m  = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
        ax.errorbar(x, m, yerr=se, fmt='o', color=col,
                    markersize=8, capsize=4, lw=2, zorder=5)

    # Wilcoxon signed-rank test
    _, pval = wilcoxon(up_vals, down_vals)
    ymax = max(up_vals.max(), down_vals.max())
    ypad = (ymax - min(up_vals.min(), down_vals.min())) * 0.08
    ytop = ymax + ypad
    ax.plot([0, 0, 1, 1], [ytop * 0.97, ytop, ytop, ytop * 0.97], lw=1.2, color='k')
    pstr = f'p = {pval:.3g}' if pval >= 0.001 else 'p < 0.001'
    ax.text(0.5, ytop * 1.01, pstr, ha='center', va='bottom', fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Up', 'Down'], fontsize=11)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel('Firing rate (Hz)', fontsize=11)
    ax.set_title(f'{name.upper()}  (n = {len(up_vals)})', fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

# --- Panel 3 : delta comparison ADN vs LMN ---
ax = axes[2]
adn_delta = frates['adn']['delta'].values
lmn_delta = frates['lmn']['delta'].values

jitter = 0.08
np.random.seed(42)
ax.scatter(np.random.normal(0, jitter, len(adn_delta)), adn_delta,
           color=col_adn, alpha=0.4, s=15, zorder=2)
ax.scatter(np.random.normal(1, jitter, len(lmn_delta)), lmn_delta,
           color=col_lmn, alpha=0.4, s=15, zorder=2)

means = []
for x, vals, c in [(0, adn_delta, col_adn), (1, lmn_delta, col_lmn)]:
    m  = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    means.append(m)
    ax.errorbar(x, m, yerr=se, fmt='o', color=c,
                markersize=9, capsize=5, lw=2.5, zorder=5)
ax.plot([0, 1], means, '-', color='k', lw=1.5, zorder=4)

ax.axhline(0, color='k', lw=1, ls='--', alpha=0.5)

_, pval = mannwhitneyu(adn_delta, lmn_delta, alternative='two-sided')
ymax = max(adn_delta.max(), lmn_delta.max())
ypad = (ymax - min(adn_delta.min(), lmn_delta.min())) * 0.08
ytop = ymax + ypad
ax.plot([0, 0, 1, 1], [ytop * 0.97, ytop, ytop, ytop * 0.97], lw=1.2, color='k')
pstr = f'p = {pval:.3g}' if pval >= 0.001 else 'p < 0.001'
ax.text(0.5, ytop * 1.01, pstr, ha='center', va='bottom', fontsize=9)

ax.set_xticks([0, 1])
ax.set_xticklabels(['ADN', 'LMN'], fontsize=11)
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel('Modulation index\n(down − up) / up', fontsize=11)
ax.set_title('Delta comparison', fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('ADN vs LMN — UP / DOWN state modulation', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/LMN_ADN_UP_DOWN_review.png"),
    dpi=300, bbox_inches='tight'
)

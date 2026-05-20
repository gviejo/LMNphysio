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
from scipy.stats import zscore, wilcoxon



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
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)



frates = []


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

        spikes = spikes[spikes.location == "lmn"]        


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
            up_rate = spikes.restrict(up_ep).rate
            down_rate = spikes.restrict(down_ep).rate
            # delta = (down_rate - up_rate) / down_rate
            delta = (down_rate - up_rate) / up_rate

            tmp = pd.DataFrame({
                "up": up_rate,
                "down": down_rate,
                "delta": delta,
                })
            tmp.index = [s.split("/")[-1] +"_"+str(i) for i in tmp.index]

            frates.append(tmp)



frates = pd.concat(frates, axis = 0)

cPickle.dump(
    {'frates': frates},
    open(os.path.expanduser("~/Dropbox/LMNphysio/data/LMN_UP_DOWN.pickle"), 'wb')
)

# %%
# --------------------------
# Figure
# --------------------------
col_up   = '#4C9BE8'   # blue  — up state
col_down = '#E8714C'   # orange — down state

stat, pval = wilcoxon(frates['up'], frates['down'])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.4)

# --- Panel 1 : paired rates ---
ax = axes[0]
up_vals   = frates['up'].values
down_vals = frates['down'].values

for u, d in zip(up_vals, down_vals):
    ax.plot([0, 1], [u, d], '-', color='grey', alpha=0.3, lw=0.8)

ax.plot([0, 1], [up_vals.mean(), down_vals.mean()], '-', color='k', lw=2, zorder=4)

# mean ± SEM overlay
for x, vals, c in [(0, up_vals, col_up), (1, down_vals, col_down)]:
    m  = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    ax.errorbar(x, m, yerr=se, fmt='o', color=c,
                markersize=8, capsize=4, lw=2, zorder=5)

# significance bracket
ymax = max(up_vals.max(), down_vals.max()) * 1.1
ax.plot([0, 0, 1, 1], [ymax * 0.97, ymax, ymax, ymax * 0.97],
        lw=1.2, color='k')
pstr = f'p = {pval:.3g}' if pval >= 0.001 else 'p < 0.001'
ax.text(0.5, ymax * 1.02, pstr, ha='center', va='bottom', fontsize=9)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Up', 'Down'], fontsize=11)
ax.set_xlim(-0.4, 1.4)
ax.set_ylabel('Firing rate (Hz)', fontsize=11)
ax.set_title(f'LMN firing rate\n(n = {len(frates)} neurons)', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

# --- Panel 2 : delta histogram ---
ax = axes[1]
bins = np.linspace(frates['delta'].min(), frates['delta'].max(), 25)
ax.hist(frates['delta'], bins=bins,
        color='grey',   alpha=0.8, label='Delta index')
ax.axvline(0, color='k', lw=1.2, ls='--')
ax.axvline(np.median(frates['delta']), color='red', lw=1.5, ls='-',
           label=f'median = {np.median(frates["delta"]):.2f}')
ax.set_xlabel('Modulation index\n(down − up) / down', fontsize=11)
ax.set_ylabel('Neuron count', fontsize=11)
ax.set_title('Firing rate modulation\nduring DOWN vs UP states', fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('LMN neurons — UP / DOWN state modulation', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/LMN_UP_DOWN_review.png"),
    dpi=300, bbox_inches='tight'
)
# plt.show()

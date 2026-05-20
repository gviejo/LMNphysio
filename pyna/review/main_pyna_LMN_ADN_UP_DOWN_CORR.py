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


allr = []
pearson = {}


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


        spikes = spikes[tokeep]
        tcurves         = tuning_curves[tokeep]

        try:
            velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
        except:
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)


        lmn = spikes.index[spikes.location == 'lmn']
        spikes = spikes[lmn]
        if len(spikes) > 4:
            ###############################################################################################
            # PEARSON CORRELATION
            ###############################################################################################
            rates = {}
            for e, ep, bin_size, std in zip(['wak', 'up', 'down'], [newwake_ep, up_ep, down_ep], [0.1, 0.02, 0.02],
                                            [3, 3, 3]):
                ep = ep.drop_short_intervals(bin_size * 22)
                count = spikes.count(bin_size, ep)
                rate = count / bin_size
                # rate = rate.as_dataframe()
                rate = rate.smooth(std=bin_size * std, windowsize=bin_size * 20).as_dataframe()
                rate = rate.apply(zscore)
                rates[e] = rate

            # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
            pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))
            pairs = pd.MultiIndex.from_tuples(pairs)
            r = pd.DataFrame(index=pairs, columns=rates.keys(), dtype=np.float32)

            for ep in rates.keys():
                tmp = np.corrcoef(rates[ep].values.T)
                r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

            name = basename
            pairs = list(combinations([name + '_' + str(n) for n in spikes.keys()], 2))
            pairs = pd.MultiIndex.from_tuples(pairs)
            r.index = pairs

            #######################
            # COMPUTING PEARSON R FOR EACH SESSION
            #######################
            pearson[s] = np.zeros((3)) * np.nan
            pearson[s][0] = scipy.stats.pearsonr(r['wak'], r['up'])[0]
            pearson[s][1] = scipy.stats.pearsonr(r['wak'], r['down'])[0]
            pearson[s][-1] = len(spikes)

            #######################
            # SAVING
            #######################
            allr.append(r)


allr = pd.concat(allr)
pearson = pd.DataFrame(pearson).T
pearson.columns = ['up', 'down', 'count']

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
})

colors = {'up': '#4477AA', 'down': '#CC3311'}

def fmt_p(p):
    if p < 0.001:
        return 'p < 0.001'
    elif p < 0.01:
        return f'p = {p:.3f}'
    else:
        return f'p = {p:.2f}'

fig = plt.figure(figsize=(12, 4))
gs = GridSpec(1, 3, figure=fig, wspace=0.4)

for i, e in enumerate(['up', 'down']):
    ax = fig.add_subplot(gs[0, i])
    tmp = allr[['wak', e]].dropna()

    ax.scatter(tmp['wak'], tmp[e], s=6, alpha=0.25, color=colors[e], linewidths=0)

    m, b = np.polyfit(tmp['wak'].values, tmp[e].values, 1)
    xr = np.linspace(tmp['wak'].min(), tmp['wak'].max(), 100)
    ax.plot(xr, m * xr + b, color=colors[e], lw=2, zorder=5)

    lims = [
        min(tmp['wak'].min(), tmp[e].min()) - 0.05,
        max(tmp['wak'].max(), tmp[e].max()) + 0.05,
    ]
    ax.plot(lims, lims, '--', color='gray', lw=1, alpha=0.5, zorder=1)

    r, p = scipy.stats.pearsonr(tmp['wak'], tmp[e])
    ax.set_xlabel('Wake correlation')
    ax.set_ylabel(f'{"Up" if e == "up" else "Down"} state correlation')
    ax.set_title(f'r = {r:.3f},  {fmt_p(p)}', fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

ax3 = fig.add_subplot(gs[0, 2])
np.random.seed(42)
jup = np.random.randn(len(pearson)) * 0.04
jdn = np.random.randn(len(pearson)) * 0.04

valid = pearson[['up', 'down']].dropna()
for idx in valid.index:
    j = valid.index.get_loc(idx)
    ax3.plot(
        [0 + jup[j], 1 + jdn[j]],
        [valid.loc[idx, 'up'], valid.loc[idx, 'down']],
        color='gray', lw=0.8, alpha=0.35, zorder=1,
    )

ax3.scatter(np.zeros(len(valid)) + jup[:len(valid)], valid['up'].values,
            s=40, color=colors['up'], alpha=0.8, zorder=3)
ax3.scatter(np.ones(len(valid)) + jdn[:len(valid)], valid['down'].values,
            s=40, color=colors['down'], alpha=0.8, zorder=3)

ax3.plot([-0.15, 0.15], [valid['up'].mean()] * 2, color=colors['up'], lw=3, zorder=4)
ax3.plot([0.85, 1.15], [valid['down'].mean()] * 2, color=colors['down'], lw=3, zorder=4)

stat_w, p_wilcox = scipy.stats.wilcoxon(valid['up'], valid['down'])
print(f"Wilcoxon up vs down: stat={stat_w:.3f}, {fmt_p(p_wilcox)}")
print(scipy.stats.ttest_rel(valid['up'], valid['down']))

ax3.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.4)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Up', 'Down'])
ax3.set_ylabel('Pearson r (with wake)')
ax3.set_xlim(-0.5, 1.5)
ax3.set_ylim(-0.2, 1.05)
ax3.set_title(f'Wilcoxon {fmt_p(p_wilcox)}', fontsize=10)

plt.tight_layout()
plt.savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/fig_LMN_UP_DOWN_CORR.png"),
    dpi=300, bbox_inches='tight'
)



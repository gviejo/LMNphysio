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



lmn_cc_down = []
psb_cc_down = []
lmn_cc_up = []
psb_cc_up = []


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

        spikes2 = spikes[spikes.location == "lmn"]


        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(
            spikes2, position['ry'], 120, minmax=(0, 2*np.pi),
            ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        
        SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
        spikes2.set_info(SI)

        spikes2 = spikes2[spikes2.SI>0.1]

        # CHECKING HALF EPOCHS
        wake2_ep = splitWake(position.time_support.loc[[0]])    
        tokeep2 = []
        stats2 = []
        tcurves2 = []   
        for i in range(2):
            tcurves_half = nap.compute_1d_tuning_curves(
                spikes2, position['ry'], 120, minmax=(0, 2*np.pi),
                ep = wake2_ep[i]
                )
            tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

            tokeep, stat = findHDCells(tcurves_half)
            tokeep2.append(tokeep)
            stats2.append(stat)
            tcurves2.append(tcurves_half)       
        lmn = np.intersect1d(tokeep2[0], tokeep2[1])
    
        if len(lmn) > 4:

            lmn_mua = spikes[lmn].to_tsd()
            psb_mua = spikes[spikes.location == "psb"].to_tsd()

            mua = nap.TsGroup({0:psb_mua, 1:lmn_mua}, metadata={"location": ["psb", "lmn"]})

            cc = nap.compute_eventcorrelogram(mua, nap.Ts(down_ep.start), 0.001, 0.2, norm=True)
            lmn_cc_down.append(cc[1])
            psb_cc_down.append(cc[0])

            cc = nap.compute_eventcorrelogram(mua, nap.Ts(up_ep.start), 0.001, 0.2, norm=True)
            lmn_cc_up.append(cc[1])
            psb_cc_up.append(cc[0])


###############################################################################################
# STACK
###############################################################################################
lmn_cc_down = pd.concat(lmn_cc_down, axis=1)
psb_cc_down = pd.concat(psb_cc_down, axis=1)
lmn_cc_up   = pd.concat(lmn_cc_up,   axis=1)
psb_cc_up   = pd.concat(psb_cc_up,   axis=1)

lags = lmn_cc_down.index.values * 1000  # ms

###############################################################################################
# SAVE
###############################################################################################
cPickle.dump({
    'lmn_cc_down': lmn_cc_down,
    'psb_cc_down': psb_cc_down,
    'lmn_cc_up':   lmn_cc_up,
    'psb_cc_up':   psb_cc_up,
    'lags': lags
}, open(os.path.expanduser("~/Dropbox/LMNphysio/data/CC_LMN_DOWN.pickle"), 'wb'))


###############################################################################################
# FIGURE  — 2 rows (cc | derivative) × 2 columns (DOWN onset | UP onset)
###############################################################################################
col_psb = '#70AD47'   # green  — PSB
col_lmn = '#ED7D31'   # orange — LMN

from scipy.ndimage import gaussian_filter1d
_smooth = lambda x: gaussian_filter1d(x, sigma=5)   # sigma in samples; bin=1ms → ~5ms smoothing

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.subplots_adjust(hspace=0.5, wspace=0.4)

def _plot_cc(ax, data, color, label, event_label):
    m = data.mean(axis=1).values
    s = data.sem(axis=1).values
    ax.fill_between(lags, m - s, m + s, color=color, alpha=0.25)
    ax.plot(lags, m, color=color, lw=2, label=label)
    ax.axvline(0, color='k', lw=1, ls='--', alpha=0.6)
    ax.axhline(1, color='k', lw=0.8, ls=':', alpha=0.4)
    ax.set_xlabel(f'Time from {event_label} (ms)')
    ax.set_ylabel('Norm. firing rate')
    ax.spines[['top', 'right']].set_visible(False)

def _plot_deriv(ax, data, color, label, event_label):
    m = _smooth(data.mean(axis=1).values)
    dm = np.gradient(m, lags)
    dm = (dm - dm.mean()) / dm.std()
    # propagate SEM through smoothing then gradient then z-score scaling
    s = _smooth(data.sem(axis=1).values)
    ds = np.abs(np.gradient(s, lags)) / dm.std()
    ax.fill_between(lags, dm - ds, dm + ds, color=color, alpha=0.25)
    ax.plot(lags, dm, color=color, lw=2, label=label)
    ax.axvline(0, color='k', lw=1, ls='--', alpha=0.6)
    ax.axhline(0, color='k', lw=0.8, ls=':', alpha=0.4)
    ax.set_xlabel(f'Time from {event_label} (ms)')
    ax.set_ylabel('z-score derivative')
    ax.spines[['top', 'right']].set_visible(False)

for col, (event_label, psb_cc, lmn_cc) in enumerate([
        ('DOWN onset', psb_cc_down, lmn_cc_down),
        ('UP onset',   psb_cc_up,   lmn_cc_up),
]):
    # Row 0 : raw cc
    _plot_cc(axes[0, col], psb_cc, col_psb, 'PSB', event_label)
    _plot_cc(axes[0, col], lmn_cc, col_lmn, 'LMN', event_label)
    axes[0, col].legend(frameon=False)
    axes[0, col].set_title(f'PSB vs LMN — {event_label}')

    # Row 1 : derivative
    _plot_deriv(axes[1, col], psb_cc, col_psb, 'PSB', event_label)
    _plot_deriv(axes[1, col], lmn_cc, col_lmn, 'LMN', event_label)
    axes[1, col].legend(frameon=False)
    axes[1, col].set_title(f'Derivative — {event_label}')

plt.savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/cc_LMN_DOWN.png"),
    dpi=150, bbox_inches='tight'
)
plt.show()

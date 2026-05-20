"""
"""

# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-29 13:46:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-29 17:14:58
# %%
import numpy as np
import pandas as pd
import pynapple as nap
# import nwbmatic as ntm
from pylab import *
import sys, os
try:
    from functions import *
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    from functions import *
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from scipy.stats import zscore, wilcoxon, mannwhitneyu
from scipy.ndimage import gaussian_filter1d
import yaml
import warnings
warnings.simplefilter('ignore', category=UserWarning)



# %%
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

datasets = yaml.safe_load(open("/mnt/ceph/users/gviejo/datasets_OPTO.yaml", "r"))
datasets = datasets['opto']['psb']

SI_thr = {
    'adn':0.2,
    'lmn':0.1,
    'psb':1.0
    }

allr = {}

# %%
for gr in ['opto', 'ctrl']:

    allr[gr] = {}

    for sd in ['ipsi']:

        allr[gr][sd] = {}

        for ep in ['wake', 'sleep']:

            dataset = datasets[gr][sd][ep]
            allr[gr][sd][ep] = []

            if len(dataset):

                for s in dataset:
                    print(s)
                    ###############################################################################################
                    # LOADING DATA
                    ###############################################################################################
                    path = os.path.join(data_directory, "OPTO", s)
                    basename = os.path.basename(path)
                    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

                    if not os.path.exists(filepath):
                        print("missing ", gr, sd, s)
                        continue
                    else:

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
                        opto_ep = nwb["opto"]

                    spikes = spikes[spikes.location == "psb"]
                    if ep == "wake":
                        stim_duration = 10.0
                    else:
                        stim_duration = 1.0

                    opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start']) >= stim_duration - 0.001]

                    ###############################################################################################
                    # COMPUTING TUNING CURVES
                    ###############################################################################################
                    # Opto ep is always the second wake
                    assert len(opto_ep.intersect(position.time_support[0])) == 0

                    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
                                                                 ep=position.time_support.loc[[0]])
                    tuning_curves = smoothAngularTuningCurves(tuning_curves)
                    tcurves = tuning_curves
                    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2 * np.pi))
                    spikes.set_info(SI)
                    spikes.set_info(max_fr=tcurves.max())

                    # figure()
                    # for i in range(len(spikes)):
                    #     subplot(6,6,i+1, projection='polar')
                    #     plot(tuning_curves[spikes.keys()[i]])
                    #     title(np.round(spikes.SI[spikes.keys()[i]], 3))
                    # show()

                    spikes = spikes.getby_threshold("SI", SI_thr["psb"])
                    spikes = spikes.getby_threshold("rate", 1.0)
                    spikes = spikes.getby_threshold("max_fr", 3.0)

                    tokeep = spikes.index
                    tcurves = tcurves[tokeep]
                    tuning_curves = tcurves
                    peaks = pd.Series(index=tcurves.columns, data=np.array(
                        [circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
                    order = np.argsort(peaks.values)
                    spikes.set_info(order=order, peaks=peaks)
                    tuning_curves.columns = [basename + "_" + str(i) for i in tuning_curves.columns]

                    ###############################################################################################
                    # PEARSON CORRELATION
                    ###############################################################################################
                    if len(tokeep) > 3:

                        velocity = np.abs(computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2))
                        newwake_ep = velocity.threshold(0.02).time_support.drop_short_intervals(1).merge_close_intervals(1)

                        pre_ep = nap.IntervalSet(start=opto_ep.start - stim_duration, end=opto_ep.start)
                        # pre_ep = pre_ep.intersect(sws_ep)
                        pre_ep = pre_ep.drop_short_intervals(stim_duration - 0.001)

                        rates = {}
                        for e, iset, bin_size, std in zip(['wak', 'pre', 'opto'], [newwake_ep, pre_ep, opto_ep], [0.1, 0.1, 0.1],
                                                          [3, 3, 3]):
                            count = spikes.count(bin_size, iset)
                            rate = count / bin_size
                            rate = rate.smooth(std=bin_size * std).as_dataframe()
                            rate = rate.apply(zscore)
                            rates[e] = rate

                        pairs = [basename + "_" + i + "-" + j for i, j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
                        r = pd.DataFrame(index=pairs, columns=rates.keys(), dtype=np.float32)

                        for e in rates.keys():
                            tmp = np.corrcoef(rates[e].values.T)
                            if len(tmp):
                                r[e] = tmp[np.triu_indices(tmp.shape[0], 1)]

                        allr[gr][sd][ep].append(r)

            allr[gr][sd][ep] = pd.concat(allr[gr][sd][ep], axis=0)


##########################################################################
# PLOTTING
##########################################################################
import scipy

fontsize = 10
COLOR = (0.25, 0.25, 0.25)

rcParams["font.family"] = 'sans-serif'
rcParams["font.size"] = fontsize
rcParams["text.color"] = COLOR
rcParams["axes.labelcolor"] = COLOR
rcParams["axes.labelsize"] = fontsize
rcParams["axes.labelpad"] = 1
rcParams["axes.titlesize"] = fontsize + 1
rcParams["xtick.labelsize"] = fontsize
rcParams["ytick.labelsize"] = fontsize
rcParams["legend.fontsize"] = fontsize
rcParams["figure.titlesize"] = fontsize
rcParams["xtick.major.size"] = 2.0
rcParams["ytick.major.size"] = 2.0
rcParams["xtick.major.width"] = 0.4
rcParams["ytick.major.width"] = 0.4
rcParams["axes.linewidth"] = 0.6
rcParams["axes.edgecolor"] = COLOR
rcParams["axes.axisbelow"] = True
rcParams["xtick.color"] = COLOR
rcParams["ytick.color"] = COLOR
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 3))
    ax.spines['bottom'].set_position(('outward', 2))


color_pre = "#7BAFD4"
color_opto = "#F4845F"
ep_labels = {'sleep': 'Sleep', 'wake': 'Wake'}
cond_colors = {'pre': color_pre, 'opto': color_opto}
cond_labels = {'pre': 'Pre-opto', 'opto': 'Opto'}

fig, axes = subplots(2, 3, figsize=(12, 6))
fig.subplots_adjust(wspace=0.5, hspace=0.55, left=0.2)

for row_idx, ep in enumerate(['sleep', 'wake']):
    tmp = allr['opto']['ipsi'][ep].dropna()

    print(f"{ep_labels[ep]}: n={len(tmp)} pairs, mean r (pre)={tmp['pre'].mean():.3f}, mean r (opto)={tmp['opto'].mean():.3f}")

    # Panels 0 & 1: individual scatter (wak vs pre, wak vs opto)
    for col_idx, cond in enumerate(['pre', 'opto']):
        ax = axes[row_idx, col_idx]
        col = cond_colors[cond]

        ax.scatter(tmp['wak'], tmp[cond], s=6, c=col, alpha=0.4,
                   linewidths=0, rasterized=True)

        v_min = min(tmp['wak'].min(), tmp[cond].min())
        v_max = max(tmp['wak'].max(), tmp[cond].max())
        ax.plot([v_min, v_max], [v_min, v_max], '--', color='grey',
                linewidth=0.8, alpha=0.6)

        m, b = np.polyfit(tmp['wak'], tmp[cond], 1)
        x_fit = np.linspace(tmp['wak'].min(), tmp['wak'].max(), 50)
        ax.plot(x_fit, m * x_fit + b, color=col, linewidth=1.5)

        r, p = scipy.stats.pearsonr(tmp['wak'], tmp[cond])
        p_str = ("p<0.001" if p < 0.001
                 else (f"p={p:.3f}" if p < 0.05 else f"p={p:.2f}"))
        ax.text(0.05, 0.95, f"r = {r:.2f}\n{p_str}",
                transform=ax.transAxes, fontsize=fontsize - 0.5,
                va='top', ha='left', color=col)

        simpleaxis(ax)
        ax.set_xlabel("Wake  pair correlation", labelpad=3)
        ax.set_ylabel(f"{cond_labels[cond]}\npair correlation", labelpad=3)

        if row_idx == 0:
            ax.set_title(cond_labels[cond], color=col,
                         fontweight='bold', pad=6)

    # Panel 2: overlay regression lines for direct comparison
    ax2 = axes[row_idx, 2]
    legend_handles = []
    for cond in ['pre', 'opto']:
        col = cond_colors[cond]
        ax2.scatter(tmp['wak'], tmp[cond], s=4, c=col, alpha=0.2,
                    linewidths=0, rasterized=True)
        m, b = np.polyfit(tmp['wak'], tmp[cond], 1)
        x_fit = np.linspace(tmp['wak'].min(), tmp['wak'].max(), 50)
        line, = ax2.plot(x_fit, m * x_fit + b, color=col, linewidth=1.5,
                         label=cond_labels[cond])
        legend_handles.append(line)

    v_min = min(tmp['wak'].min(), tmp['pre'].min(), tmp['opto'].min())
    v_max = max(tmp['wak'].max(), tmp['pre'].max(), tmp['opto'].max())
    ax2.plot([v_min, v_max], [v_min, v_max], '--', color='grey',
             linewidth=0.8, alpha=0.6)

    simpleaxis(ax2)
    ax2.set_xlabel("Wake  pair correlation", labelpad=3)
    ax2.set_ylabel("Pair correlation", labelpad=3)

    if row_idx == 0:
        ax2.set_title("Comparison", fontweight='bold', pad=6)
        ax2.legend(handles=legend_handles, frameon=False, handlelength=1.5,
                   loc='upper left', borderpad=0, labelspacing=0.3)

    # Row label
    axes[row_idx, 0].text(
        -0.42, 0.5, ep_labels[ep],
        transform=axes[row_idx, 0].transAxes,
        fontsize=fontsize + 1, fontweight='bold',
        va='center', ha='center', rotation=90
    )

fig.suptitle("PSB population coherence  –  Opto group (ipsi)",
             fontsize=fontsize + 2, y=1.02)



savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/fig_opto_psb_coherence.png"),
    dpi=200, bbox_inches='tight'
)

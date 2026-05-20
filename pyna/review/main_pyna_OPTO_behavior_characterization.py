"""
To characterize the behavior of the animals during the opto sessions, and see if it differs between groups and conditions.
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

datasets = yaml.safe_load(
    open(os.path.join(
        data_directory,
        "datasets_OPTO.yaml"), "r"))['opto']


SI_thr = {
    'adn':0.2,
    'lmn':0.1,
    'psb':1.0
    }

# %%
records = []  # one row per session

cc_linvel = {}
cc_angvel = {}

for st in ['adn', 'lmn']:

    cc_linvel[st] = {}
    cc_angvel[st] = {}

    for gr in ['opto', 'ctrl']:

        cc_linvel[st][gr] = {}
        cc_angvel[st][gr] = {}

        for sd in ['ipsi', 'bilateral']:

            cc_linvel[st][gr][sd] = []
            cc_angvel[st][gr][sd] = []

            try:
                dataset = datasets[st][gr][sd]['wake']
            except:
                dataset = []

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
                        print("missing ", st, gr, s)
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

                    spikes = spikes[spikes.location == st]
                    stim_duration = 10.0
                    opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start']) >= stim_duration - 0.001]

                    bin_size = 0.05

                    angular_velocity = np.abs((np.unwrap(position['ry']).bin_average(bin_size).derivative()) % (2 * np.pi))

                    pos = position[['x', 'z']].bin_average(bin_size)
                    pos_smooth = pos.values.copy()

                    pos_smooth[:, 0] = gaussian_filter1d(pos_smooth[:, 0], sigma=1)
                    pos_smooth[:, 1] = gaussian_filter1d(pos_smooth[:, 1], sigma=1)
                    dt           = np.diff(pos.t)
                    displacement = np.sqrt(np.sum(np.diff(pos_smooth, axis=0)**2, axis=1))
                    linear_velocity = nap.Tsd(t=pos.t[:-1], d=displacement / dt,
                                             time_support=position.time_support)

                    # -------------------------------------------------------
                    # Velocity during opto ON vs opto OFF (rest of wake)
                    # -------------------------------------------------------
                    wake_full = position.time_support

                    baseline_ep = wake_full.set_diff(opto_ep)

                    ang_on  = float(angular_velocity.restrict(opto_ep).mean())
                    ang_off = float(angular_velocity.restrict(baseline_ep).mean())
                    lin_on  = float(linear_velocity.restrict(opto_ep).mean())
                    lin_off = float(linear_velocity.restrict(baseline_ep).mean())

                    records.append({
                        'session':  s,
                        'area':     st,
                        'group':    gr,
                        'side':     sd,
                        'ang_on':   ang_on,
                        'ang_off':  ang_off,
                        'lin_on':   lin_on,
                        'lin_off':  lin_off,
                    })

                    ###################################################
                    # tEMPORAL CORRELATION OF VELOCITY WITH STIMULATION
                    ##################################################
                    cc_linvel[st][gr][sd].append(
                        np.mean(nap.compute_perievent(linear_velocity, nap.Ts(opto_ep.start), (-5, 15)), 1).as_series()
                    )
                    cc_angvel[st][gr][sd].append(
                        np.mean(nap.compute_perievent(angular_velocity, nap.Ts(opto_ep.start), (-5, 15)), 1).as_series()
                    )


for st in ['adn', 'lmn']:

    for gr in ['opto', 'ctrl']:

        for sd in ['ipsi', 'bilateral']:

            if len(cc_linvel[st][gr][sd]) > 0:
                cc_linvel[st][gr][sd] = pd.concat(cc_linvel[st][gr][sd], axis=1)
            else:
                cc_linvel[st][gr][sd] = None

            if len(cc_angvel[st][gr][sd]) > 0:
                cc_angvel[st][gr][sd] = pd.concat(cc_angvel[st][gr][sd], axis=1)
            else:
                cc_angvel[st][gr][sd] = None

cc_all = {}
# Merging lmn and adn
for gr in ['opto', 'ctrl']:
    cc_all[gr] = {}
    sd = "ipsi" # concatenating adn and lmn
    cc_all[gr][sd] = {}
    cc_all[gr][sd]["lin"] = pd.concat([cc_linvel['adn'][gr][sd], cc_linvel['lmn'][gr][sd]], axis=1)
    cc_all[gr][sd]["ang"] = pd.concat([cc_angvel['adn'][gr][sd], cc_angvel['lmn'][gr][sd]], axis=1)
    sd = "bilateral" # only adn
    cc_all[gr][sd] = {}
    cc_all[gr][sd]["lin"] = cc_linvel['adn'][gr][sd]
    cc_all[gr][sd]["ang"] = cc_angvel['adn'][gr][sd]



# %%
###############################################################################################
# FIGURE — opto ON vs OFF (row 0) + peri-event traces (row 1)
###############################################################################################
df = pd.DataFrame(records)

cPickle.dump(
    {'df': df, 'cc_all': cc_all},
    open(os.path.expanduser("~/Dropbox/LMNphysio/data/OPTO_behavior_characterization.pickle"), 'wb')
)

col_ipsi  = '#5B8DB8'   # blue  — ipsi
col_bi    = '#E07B39'   # orange — bilateral

metrics_pairs = [
    ('ang_off', 'ang_on', 'Angular velocity (rad/s)', 'ang'),
    ('lin_off', 'lin_on', 'Linear velocity (m/s)',    'lin'),
]

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.5)

for col_idx, (col_off, col_on, ylabel, cc_key) in enumerate(metrics_pairs):

    # ── row 0: ON vs OFF paired plot ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, col_idx])

    for x_off, sd, col in [(0, 'ipsi', col_ipsi), (2, 'bilateral', col_bi)]:
        sub = df[(df['side'] == sd) & (df['group'] == 'opto')].dropna(subset=[col_on, col_off])
        on_vals  = sub[col_on].values
        off_vals = sub[col_off].values
        if len(on_vals) < 2:
            continue

        ax.plot([x_off, x_off + 1], [off_vals, on_vals],
                '-', color=col, alpha=0.35, lw=0.9)

        for x, vals in [(x_off, off_vals), (x_off + 1, on_vals)]:
            m  = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            ax.errorbar(x, m, yerr=se, fmt='o', color=col,
                        markersize=8, capsize=4, lw=2, zorder=5)
        ax.plot([x_off, x_off + 1], [off_vals.mean(), on_vals.mean()],
                '-', color=col, lw=2, zorder=4)

        _, pval = wilcoxon(off_vals, on_vals)
        ymax = max(on_vals.max(), off_vals.max())
        ytop = ymax + ymax * 0.08
        ax.plot([x_off, x_off, x_off + 1, x_off + 1],
                [ytop * 0.97, ytop, ytop, ytop * 0.97], lw=1.2, color=col)
        pstr = f'p={pval:.3g}' if pval >= 0.001 else 'p<0.001'
        ax.text(x_off + 0.5, ytop * 1.01, pstr,
                ha='center', va='bottom', fontsize=8, color=col)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['OFF', 'ON', 'OFF', 'ON'], fontsize=10)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ybot, ytop_ax = ax.get_ylim()
    yspan = ytop_ax - ybot
    for x_mid, label, col in [(0.5, 'Ipsi', col_ipsi), (2.5, 'Bilateral', col_bi)]:
        ax.text(x_mid, ybot - yspan * 0.14, label,
                ha='center', va='top', fontsize=10, color=col, fontweight='bold')

    # ── row 1: peri-event traces ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, col_idx])

    for sd, col, ls in [('ipsi', col_ipsi, '-'), ('bilateral', col_bi, '--')]:
        data = cc_all['opto'][sd][cc_key]
        if data is None:
            continue
        t    = data.index.values
        mean = data.mean(axis=1).values
        sem  = data.std(axis=1).values / np.sqrt(data.shape[1])
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.2, color=col)
        ax.plot(t, mean, ls=ls, color=col, lw=1.8, label=sd.capitalize())

    ax.axvline(0,  color='k', lw=1, ls='--', alpha=0.5)
    ax.axvline(10, color='k', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel('Time from stimulation (s)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

plt.suptitle('Optogenetic stimulation — effect on locomotion (opto group)', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/OPTO_behavior_characterization.png"),
    dpi=300, bbox_inches='tight'
)

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-08-06 10:02:46
import numpy as np
import pandas as pd
import pynapple as nap

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


from scipy.stats import zscore, wilcoxon

# matplotlib.style.use('seaborn-paper')
import matplotlib.image as mpimg

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
# import hsluv

import os
import sys

from scipy.ndimage import gaussian_filter

try:
    from functions import *
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    from functions import *


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
else:
    data_directory = "~"




def figsize(scale):
    fig_width_pt = 483.69687  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2  # Aesthetic ratio (you could change this)
    # fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_width = 6
    fig_height = fig_width * golden_mean * 0.5  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    gca().spines['left'].set_position(('outward', 3))
    gca().spines['bottom'].set_position(('outward', 2))

    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)


def noaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)


# font_dir = [os.path.expanduser("~")+'/Dropbox/CosyneData/figures_poster_2022']
# for font in font_manager.findSystemFonts(font_dir):
#     font_manager.fontManager.addfont(font)

fontsize = 6.0

COLOR = (0.25, 0.25, 0.25)
cycle = rcParams['axes.prop_cycle'].by_key()['color'][5:]

rcParams["font.family"] = 'sans-serif'
rcParams["font.size"] = fontsize
rcParams["text.color"] = COLOR
rcParams["axes.labelcolor"] = COLOR
rcParams["axes.labelsize"] = fontsize
rcParams["axes.labelpad"] = 1
# rcParams['axes.labelweight'] = 'bold'
rcParams["axes.titlesize"] = fontsize
rcParams["xtick.labelsize"] = fontsize
rcParams["ytick.labelsize"] = fontsize
rcParams["legend.fontsize"] = fontsize
rcParams["figure.titlesize"] = fontsize
rcParams["xtick.major.size"] = 1.3
rcParams["ytick.major.size"] = 1.3
rcParams["xtick.major.width"] = 0.4
rcParams["ytick.major.width"] = 0.4
rcParams["axes.linewidth"] = 0.4
rcParams["axes.edgecolor"] = COLOR
rcParams["axes.axisbelow"] = True
rcParams["xtick.color"] = COLOR
rcParams["ytick.color"] = COLOR
rcParams['xtick.major.pad'] = 0.5
rcParams['ytick.major.pad'] = 0.5
rcParams['xtick.minor.pad'] = 0.5
rcParams['ytick.minor.pad'] = 0.5

colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}
opto_color = "#DC143C"

cmap = plt.get_cmap("Set2")

# COmputing tests
map_significance = {
    1:"n.s.",
    2:"*",
    3:"**",
    4:"***"
}


###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"




###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(1, 2, hspace=0.55, wspace=0.3)

###############################################################################################
# LOAD DATA
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'OPTO_behavior_characterization.pickle'), 'rb'))
df     = data['df']
cc_all = data['cc_all']

col_ipsi = '#5B8DB8'
col_bi   = '#E07B39'

metrics_pairs = [
    ('ang_off', 'ang_on', 'Angular velocity (rad/s)', 'ang'),
    ('lin_off', 'lin_on', 'Linear velocity (m/s)',    'lin'),
]

###############################################################################################
# ROW 0 — opto ON vs OFF paired plots
###############################################################################################
gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, outergs[0], wspace=0.6)
gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, outergs[1], wspace=0.6)
gs_list = [gs1, gs2]
for col_idx, (col_off, col_on, ylabel, cc_key) in enumerate(metrics_pairs):

    ax = fig.add_subplot(gs_list[col_idx][0])
    simpleaxis(ax)

    for x_off, sd, col in [(0, 'ipsi', col_ipsi), (2, 'bilateral', col_bi)]:
        sub = df[(df['side'] == sd) & (df['group'] == 'opto')].dropna(subset=[col_on, col_off])
        on_vals  = sub[col_on].values
        off_vals = sub[col_off].values
        if len(on_vals) < 2:
            continue

        ax.plot([x_off, x_off + 1], [off_vals, on_vals],
                '-', color=col, alpha=0.3, lw=0.5)
        for x, vals in [(x_off, off_vals), (x_off + 1, on_vals)]:
            m  = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            ax.errorbar(x, m, yerr=se, fmt='o', color=col,
                        markersize=3, capsize=2, lw=1, zorder=5)
        ax.plot([x_off, x_off + 1], [off_vals.mean(), on_vals.mean()],
                '-', color=col, lw=1, zorder=4)

        z, pval = wilcoxon(off_vals, on_vals)
        print(f"{sd} — {cc_key}: z={z:.2f}, p={pval:.3g}")
        ymax = max(on_vals.max(), off_vals.max())
        ytop = ymax + ymax * 0.08
        ax.plot([x_off, x_off, x_off + 1, x_off + 1],
                [ytop * 0.97, ytop, ytop, ytop * 0.97], lw=0.8, color=col)
        pstr = f'p={pval:.3g}' if pval >= 0.001 else 'p<0.001'
        ax.text(x_off + 0.5, ytop * 1.01, pstr,
                ha='center', va='bottom', color=col)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['OFF', 'ON', 'OFF', 'ON'])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel(ylabel)
    ybot, ytop_ax = ax.get_ylim()
    yspan = ytop_ax - ybot
    for x_mid, label, col in [(0.5, 'Ipsi', col_ipsi), (2.5, 'Bilateral', col_bi)]:
        ax.text(x_mid, ybot - yspan * 0.18, label,
                ha='center', va='top', color=col, fontweight='bold')
    ax.set_title(f"{ylabel} — Opto ON vs OFF", fontsize=fontsize, fontweight='bold', pad=10, x=1.2)

###############################################################################################
# ROW 1 — peri-event velocity traces
###############################################################################################
for col_idx, (col_off, col_on, ylabel, cc_key) in enumerate(metrics_pairs):

    ax = fig.add_subplot(gs_list[col_idx][1])
    simpleaxis(ax)

    for sd, col, ls in [('ipsi', col_ipsi, '-'), ('bilateral', col_bi, '--')]:
        data_cc = cc_all['opto'][sd][cc_key]
        if data_cc is None:
            continue
        t    = data_cc.index.values
        mean = data_cc.mean(axis=1).values
        sem  = data_cc.std(axis=1).values / np.sqrt(data_cc.shape[1])
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.2, color=col)
        ax.plot(t, mean, ls=ls, color=col, lw=1, label=sd.capitalize())

    ax.axvline(0,  color='k', lw=0.6, ls='--', alpha=0.5)
    ax.axvline(10, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.set_xlabel('Time from stimulation (s)')
    ax.set_ylabel(ylabel)
    if col_idx == 0:
        ax.set_ylim(2.5, 4)
    if col_idx == 0:
        ax.legend(frameon=False)


outergs.update(top=0.86, bottom=0.2, right=0.97, left=0.06)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2025/supp7.pdf",
    dpi=200,
    facecolor="white",
)
# show()



# corr3['chfr'] = pd.Series({s:delta[delta.index.str.contains(s)].mean() for s in corr3.index.values})


# figure()
# plot(corr3['opto'], corr3['chfr'], 'o')
# xlabel("Session correlation", fontsize=12)
# ylabel("Delta firing rate", fontsize=12)
# m, b = np.polyfit(corr3['opto'], corr3['chfr'], 1)
# x = np.linspace(corr3['opto'].min(), corr3['opto'].max(),5)
# plot(x, x*m + b)
# tau, p = scipy.stats.kendalltau(corr3['opto'], corr3['chfr'])
# title(f"tau={np.round(tau,2)}, p={np.round(p,2)}", fontsize=12)
# savefig(os.path.expanduser("~") + "/Dropbox/LMNphysio/summary_opto/fig_corr_coh_rate.png")

'''
FIRING rate modulation LMN during up and down states of PSB
Pairwise correlation of firing rate modulation between LMN and PSB during up and down states of PSB with up decimated
'''
import numpy as np
import pandas as pd
import pynapple as nap
import scipy

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

from scipy.stats import zscore
from scipy.stats import mannwhitneyu

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
except:
    sys.path.append("../")
    from functions import *



def figsize(scale):
    fig_width_pt = 483.69687  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2  # Aesthetic ratio (you could change this)
    # fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_width = 6
    fig_height = fig_width * golden_mean * 0.8 # height in inches
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

fontsize = 6

COLOR = (0.25, 0.25, 0.25)

# rcParams["font.family"] = 'Liberation Sans'
rcParams["font.family"] = 'sans-serif'
# rcParams["font.family"] = 'DejaVu Sans'
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

dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"


colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}


short_epochs = {'wak':'Wake', 'sws':'non-REM', 'rem':'REM'}

cmap = plt.get_cmap("Set2")


map_significance = {
    1:"n.s.",
    2:"*",
    3:"**",
    4:"***"
}

fig = figure(figsize=figsize(1))

outergs = GridSpec(1, 2, wspace = 0.5, width_ratios=[0.5, 1.2])



####################################
# LMN firing rate modulation UP vs DOWN   outergs[0,0]
####################################
ud_data = cPickle.load(open(os.path.join(dropbox_path, 'LMN_UP_DOWN.pickle'), 'rb'))
frates = ud_data['frates']

col_up   = '#4C9BE8'
col_down = '#E8714C'

stat_ud, pval_ud = scipy.stats.wilcoxon(frates['up'], frates['down'])
print("Wilcoxon UP vs DOWN", stat_ud, pval_ud)

gs_ud = gridspec.GridSpecFromSubplotSpec(2, 1, outergs[0, 0], hspace=0.6)

# Panel 1 : paired rates
ax_ud = fig.add_subplot(gs_ud[0, 0])
simpleaxis(ax_ud)
up_vals   = frates['up'].values
down_vals = frates['down'].values
for u, d in zip(up_vals, down_vals):
    ax_ud.plot([0, 1], [u, d], '-', color=COLOR, alpha=0.2, lw=0.4)
ax_ud.plot([0, 1], [up_vals.mean(), down_vals.mean()], '-', color='k', lw=1, zorder=4)
for x, vals, c in [(0, up_vals, col_up), (1, down_vals, col_down)]:
    m  = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    ax_ud.errorbar(x, m, yerr=se, fmt='o', color=c, markersize=3, capsize=2, lw=1, zorder=5)
ymax = max(up_vals.max(), down_vals.max()) * 1.1
ax_ud.plot([0, 0, 1, 1], [ymax*0.97, ymax, ymax, ymax*0.97], lw=0.8, color='k')
pstr = f'p={pval_ud:.3g}' if pval_ud >= 0.001 else 'p<0.001'
ax_ud.text(0.5, ymax*1.02, pstr, ha='center', va='bottom')
ax_ud.set_xticks([0, 1])
ax_ud.set_xticklabels(['Up', 'Down'])
ax_ud.set_xlim(-0.4, 1.4)
ax_ud.set_ylabel('Firing rate (Hz)')
ax_ud.set_title(f'LMN rate (n={len(frates)})')

# Panel 2 : delta histogram
ax_dh = fig.add_subplot(gs_ud[1, 0])
simpleaxis(ax_dh)
bins = np.linspace(frates['delta'].min(), frates['delta'].max(), 25)
ax_dh.hist(frates['delta'], bins=bins, color=colors['lmn'], alpha=0.8)
ax_dh.axvline(0, color='k', lw=0.8, ls='--')
ax_dh.axvline(np.median(frates['delta']), color=col_down, lw=1, ls='-')
print(np.median(frates['delta']))
ax_dh.set_xlabel('(down − up) / up')
ax_dh.set_ylabel('Neuron count')
ax_dh.set_title('Modulation index')
ax_dh.set_xlim(-1, 0.01)
ax_dh.set_xticks([-1, -0.5, 0])

####################################
# CC LMN / PSB around DOWN and UP onsets   outergs[0,1]
####################################
cc_data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN_DOWN.pickle'), 'rb'))
lmn_cc_down = cc_data['lmn_cc_down']
psb_cc_down = cc_data['psb_cc_down']
lmn_cc_up   = cc_data['lmn_cc_up']
psb_cc_up   = cc_data['psb_cc_up']
lags = cc_data['lags']


col_psb = colors['psb']
col_lmn = colors['lmn']

gs_cc = gridspec.GridSpecFromSubplotSpec(2, 2, outergs[0, 1], hspace=0.4, wspace=0.5)

def _plot_cc_sub(ax, data_psb, data_lmn, event_label):
    for data, color, label in [(data_psb, col_psb, 'PSB'), (data_lmn, col_lmn, 'LMN')]:
        m = data.mean(axis=1).values
        s = data.sem(axis=1).values
        ax.fill_between(lags, m - s, m + s, color=color, alpha=0.25)
        ax.plot(lags, m, color=color, lw=1, label=label)
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(1, color='k', lw=0.5, ls=':', alpha=0.4)
    ax.set_xlabel(f'Time from {event_label} (ms)')
    ax.set_ylabel('Norm. rate')
    ax.spines[['top', 'right']].set_visible(False)

from scipy.ndimage import gaussian_filter1d

def _plot_deriv_sub(ax, data_psb, data_lmn, event_label):
    for data, color, label in [(data_psb, col_psb, 'PSB'), (data_lmn, col_lmn, 'LMN')]:
        m = gaussian_filter1d(data.mean(axis=1).values, sigma=5)
        dm = np.gradient(m, lags)
        dm = (dm - dm.mean()) / dm.std()
        s = np.abs(np.gradient(gaussian_filter1d(data.sem(axis=1).values, sigma=5), lags)) / dm.std()
        ax.fill_between(lags, dm - s, dm + s, color=color, alpha=0.25)
        ax.plot(lags, dm, color=color, lw=1, label=label)
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(0, color='k', lw=0.5, ls=':', alpha=0.4)
    ax.set_xlabel(f'Time from {event_label} (ms)')
    ax.set_ylabel('z-score deriv.')
    ax.spines[['top', 'right']].set_visible(False)

for col, (event_label, psb_cc, lmn_cc) in enumerate([
        ('DOWN', psb_cc_down, lmn_cc_down),
        ('UP',   psb_cc_up,   lmn_cc_up),
]):
    ax0 = fig.add_subplot(gs_cc[0, col])
    _plot_cc_sub(ax0, psb_cc, lmn_cc, event_label)
    if col == 0:
        ax0.legend(frameon=False, fontsize=5)
    ax0.set_title(f'Norm. xcorr. around {event_label} onset')

    ax1 = fig.add_subplot(gs_cc[1, col])
    _plot_deriv_sub(ax1, psb_cc, lmn_cc, event_label)
    if col == 0:
        ax1.set_title(f'z-scored derivative of xcorr.', x=1.25)


# ####################################
# # CORR LMN_PSB
# ###################################
# dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"
# data = cPickle.load(open(os.path.join(dropbox_path, 'CORR_LMN-PSB_UP_DOWN.pickle'), 'rb'))
# pearson = data['pearson']
# frates = data['frates']
# baseline = data['baseline']
#
# gs_corr_top = gridspec.GridSpecFromSubplotSpec(2,1, outergs[0,2], hspace=0.7, wspace=0.2)
#
# subplot(gs_corr_top[0,0])
# simpleaxis(gca())
#
# for s in pearson.index:
#     plot([1, 2], pearson.loc[s,['down', 'decimated']], '-', color=COLOR, linewidth=0.1)
# plot(np.ones(len(pearson)), pearson['down'], 'o', color=colors['lmn'], markersize=1)
# plot(np.ones(len(pearson))*2, pearson['decimated'], 'o', color=colors['lmn'], markersize=1)
#
# xlim(0.5, 3)
# gca().spines['bottom'].set_bounds(1, 2)
# ymin = np.minimum(pearson[['decimated','down']].min().min(), 0)
# ylim(ymin-0.1, 1.0)
# gca().spines['left'].set_bounds(ymin-0.1, 1.0)
# yticks([0, 0.5,1], [0, 0.5, 1])
# ylabel("Pop. coherence (r)", y=-0.2, labelpad=2)
# xticks([1, 2], ['Down', 'Up'])
# # title("Sessions")
#
#
# ############
# subplot(gs_corr_top[1,0])
# simpleaxis(gca())
# xlim(0.5, 3)
# ylim(-0.1, 1)
# gca().spines['bottom'].set_bounds(1, 2)
# xlabel("minus baseline", labelpad=1)
# # if i == 1: gca().spines["left"].set_visible(False)
# plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
# plot([2.2], [0], 'o', color=COLOR, markersize=0.5)
# tmp = (pearson[['down', 'decimated']]-baseline[['down', 'decimated']]).values.astype("float")
# vp = violinplot(tmp, showmeans=False,
#     showextrema=False, vert=True, side='high'
#     )
# for k, p in enumerate(vp['bodies']):
#     p.set_color(colors['lmn'])
#     p.set_alpha(1)
#
# m = [pearson[c].mean() for c in ['down', 'decimated']]
# plot([1, 2], m, 'o', markersize=0.5, color=COLOR)
#
# xticks([1,2],['',''])
# # ylabel(r"Mean$\Delta$")
#
# # COmputing tests
# map_significance = {
#     1:"n.s.",
#     2:"*",
#     3:"**",
#     4:"***"
# }
#
# for i, k in enumerate(['down', 'decimated']):
#     zw, p = scipy.stats.wilcoxon(pearson[k].values.astype("float"), baseline[k].values.astype("float"), alternative='greater')
#     signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
#     text(i+0.9, m[i]-0.07, s=map_significance[signi], va="center", ha="right")
#     print("wilcoxon baseline", zw, p, len(pearson))
#
# xl, xr = 2.5, 2.6
# plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
# plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
# plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
# zw, p = scipy.stats.wilcoxon(tmp[:,1], tmp[:,0], alternative='greater')
# signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
# text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")
#
# print("wilcoxon across", zw, p, len(tmp))



outergs.update(top=0.92, bottom=0.08, right=0.98, left=0.06)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2025/supp5.pdf",
    dpi=200,
    facecolor="white",
)

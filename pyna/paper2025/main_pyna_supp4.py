# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-07-19 15:01:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-08-04 14:35:20
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
    fig_height = fig_width * golden_mean * 0.6 # height in inches
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


colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}


short_epochs = {'wak':'Wake', 'sws':'non-REM', 'rem':'REM'}

cmap = plt.get_cmap("Set2")

###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"
data = cPickle.load(open(os.path.join(dropbox_path, 'All_TC.pickle'), 'rb'))

alltc = data['alltc']
allinfo = data['allinfo']



fig = figure(figsize=figsize(1))

outergs = GridSpec(1, 2, width_ratios=[1, 0.8], wspace=0.2)



###############
# Connected pairs
###############

# gs4 = gridspec.GridSpecFromSubplotSpec(2,1, gs3[0,1],
#     hspace=0.6
#     )

data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN-ADN.pickle'), 'rb'))

index = data['zorder'].index.values
zcc = data['zcc']
order = data['order']
allcc = data['allcc']

gs4_1 = gridspec.GridSpecFromSubplotSpec(
    5, 3, subplot_spec=outergs[0,0], hspace = 0.5, wspace=0.2
)

inds = np.indices((5, 3))

count = 0
for i, j in zip(inds[0].flat, inds[1].flat):

    gsex = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs4_1[i,j], wspace=0.1
    )

    pair = order[count]

    subplot(gsex[0,0])
    simpleaxis(gca())
    axvline(0, linewidth=0.5, color=COLOR)    

    tmp = allcc['sws'][pair].loc[-0.01:0.01]
    x = tmp.index.values
    dt = np.mean(np.diff(x))
    x = np.hstack((x-dt/2, np.array([x[-1]+dt/2])))
    # axvspan(0.002, 0.008, alpha=0.2, linewidth=0)
    stairs(tmp.values, x, fill=True, color=COLOR)

    if i != 4:
        xticks([])
    if i == 4:
        xticks([-0.01, 0, 0.01], [-10, 0, 10])
    if i==4 and j == 1:
        xlabel("Time lag (ms)")
    if i==1 and j== 0:
        ylabel("Rate (norm.)", y=0, labelpad=10)

    subplot(gsex[0,1], projection='polar')
    # simpleaxis(gca())
    for n, st in zip(pair, ['lmn', 'adn']):
        tmp =  alltc[n]
        tmp = tmp/tmp.max()
        plot(tmp, color=colors[st], linewidth=0.8)

    if i == 0 and j == 0:
        title("Norm.")
    xticks([])
    yticks([])




    count += 1


gs4_2 = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outergs[0, 1], hspace = 0.6, wspace=0.5
)

subplot(gs4_2[0,0])

simpleaxis(gca())

tmp = data['angdiff'][index]
h, b = np.histogram(tmp, np.linspace(-np.pi, np.pi, 22), range = (-np.pi, np.pi))
h = h/h.sum()*100

# stairs(h, b, fill=True, color=cmap(8), alpha=1, edgecolor=COLOR)

bar(b[0:-1]+np.diff(b)/2, h, np.diff(b).mean(), color=cmap(8))


xlim(-np.pi, np.pi)
xticks([-np.pi, 0, np.pi], [-180, 0, 180])
xlabel("Ang. diff. (deg)")
ylabel("%")

title(r"$LMN \rightarrow ADN$", pad=3)

subplot(gs4_2[0,1])

simpleaxis(gca())

angdiff = np.abs(data['angdiff'][order])
z = data['zorder'][order]

plot(angdiff.values, z.values, 'o', markersize=1, color=COLOR)

m, b = np.polyfit(angdiff.values, z.values, 1)
x = np.linspace(angdiff.values.min(), angdiff.values.max(),5)
r, p = scipy.stats.pearsonr(angdiff.values, z.values)
print(r, p)
plot(x, x*m + b, linewidth=0.5, color='red', label=f"r={np.round(r, 2)}\np<0.001")
legend(frameon=False, loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=5)
xlabel("|Ang. diff.| (deg) ")
ylabel("Z")
title("$Z_{LMN/ADN} > 3$", pad=3)


subplot(gs4_2[1,0])

simpleaxis(gca())

peak = data['zcc']['sws'][order].loc[0.001:0.008].idxmax()

hist_, bin_edges = np.histogram(peak.values, bins = np.linspace(0, 0.01, 25), range = (0, 0.01))

hist_percent = hist_ / hist_.sum() * 100

stairs(hist_percent, bin_edges, fill=True, color=COLOR, alpha=1)

xticks([0, 0.005, 0.01], [0, 5, 10])
xlabel("Lag (ms)")
ylabel("%")
title(f"mean lag = {np.round(peak.mean()*1e3, 2)} ms", pad=3)

outergs.update(top=0.93, bottom=0.11, right=0.97, left=0.08)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2025/supp3.pdf",
    dpi=200,
    facecolor="white",
)
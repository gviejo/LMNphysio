# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-07-19 15:01:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-21 18:20:22
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
    fig_height = fig_width * golden_mean * 1.1  # height in inches
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

cmap = plt.get_cmap("Set2")

###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"
data = cPickle.load(open(os.path.join(dropbox_path, 'All_TC.pickle'), 'rb'))

alltc = data['alltc']
allinfo = data['allinfo']



fig = figure(figsize=figsize(1))

outergs = GridSpec(3, 1, hspace = 0.5, height_ratios=[0.4, 0.6, 0.2])


#########################
# TUNING CURVES SELECTION
#########################

gs1 = gridspec.GridSpecFromSubplotSpec(1,2, outergs[0,0], 
    wspace = 0.5
    )

for i, st in enumerate(['adn', 'lmn']):
    gs1_1 = gridspec.GridSpecFromSubplotSpec(2,2, gs1[0,i],
        #hspace = 0.2, wspace = 0.2, width_ratios=[0.2, 0.5, 0.5])
        )

    idx = allinfo[allinfo['location'] == st].index.values

    tokeep = allinfo.loc[idx][(allinfo.loc[idx]['rate'] > 1.0) & (allinfo.loc[idx]['SI'] > 0.1) & (allinfo.loc[idx]['tokeep']==1)].index.values


    # Cloud
    ax = subplot(gs1_1[:,0])
    simpleaxis(ax)

    loglog(allinfo.loc[idx, 'rate'], allinfo.loc[idx, 'SI'], '.', markersize=1)
    loglog(allinfo.loc[tokeep, 'rate'], allinfo.loc[tokeep, 'SI'], 'o', color=colors[st], markersize=1)

    xlabel("Rate (Hz)")
    ylabel("Mutual Information\n(bits/spk)", labelpad=5)

    axvline(1.0, linewidth=0.5, color=COLOR)
    axhline(0.1, linewidth=0.5, color=COLOR)

    tc = centerTuningCurves_with_peak(alltc[tokeep])
    tc = tc/tc.max()

    # Tuning curves
    subplot(gs1_1[0,1])
    plot(tc.mean(1))

    ax = subplot(gs1_1[1,1])
    imshow(tc.values.T, aspect='auto')


################################
# PAIRWISE CORRELATION / SESSION
################################

gs2 = gridspec.GridSpecFromSubplotSpec(2,1, outergs[1,0], 
    wspace = 0.5
    )

for i, st in enumerate(['adn', 'lmn']):

    data = cPickle.load(open(os.path.join(dropbox_path, f'All_correlation_{st.upper()}.pickle'), 'rb'))
    allr = data['allr'] 
    sessions = [s.split("/")[-1] for s in data['pearsonr'].index.values]

    gs2_1 = gridspec.GridSpecFromSubplotSpec(1,len(sessions), gs2[i,0],
        #hspace = 0.2, wspace = 0.2, width_ratios=[0.2, 0.5, 0.5])
        )


    for j, s in enumerate(sessions):

        index = [k for k in allr.index.values if s in k[0]]

        subplot(gs2_1[0,j])
        # simpleaxis(gca())
        gca().set_aspect("equal")
        plot(allr.loc[index, 'wak'], allr.loc[index, 'sws'], 
               '.',                 
                color=colors[st],
                markersize=0.6 
            )
        xlim(-1, 1)
        ylim(-1,1)
        if j == 0:
            xticks([-1, 1])
            yticks([-1, 1])
        else:
            xticks([])
            yticks([])





outergs.update(top=0.96, bottom=0.06, right=0.95, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/supp1.pdf",
    dpi=200,
    facecolor="white",
)
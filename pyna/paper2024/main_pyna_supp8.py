# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-08-01 17:42:15
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


from scipy.stats import zscore, pearsonr

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


sys.path.append("../../model/")

from model import Model






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



m = Model(I_lmn = 0.0) # Sleep



###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(1, 2, hspace = 0.4, wspace=0.5, width_ratios=[0.7, 0.3])


names = {'adn':"ADN", 'lmn':"LMN"}
epochs = {'wak':'Wake', 'sws':'Sleep'}





# ##########################################
# 
# ##########################################

gs_top1 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[0,0], wspace=0.9
    )


subplot(gs_top1[0,0])

im = imshow(m.w_lmn_adn, aspect='auto', cmap='viridis', origin='lower')

xlabel("LMN")
ylabel("ADN")
title(r"$W_{LMN\rightarrow ADN}$", pad=1)

axip = gca().inset_axes([1.1, 0, 0.08, 0.5])
noaxis(axip)
cbar = colorbar(im, cax=axip)
axip.set_title("W")




subplot(gs_top1[0,1])

im = imshow(m.w_psb_lmn, aspect='auto', cmap='viridis')

xlabel("PSB (ADN)")
ylabel("LMN")
title(r"$W_{PSB\rightarrow LMN}$", pad=1)

axip = gca().inset_axes([1.1, 0, 0.08, 0.5])
noaxis(axip)
cbar = colorbar(im, cax=axip, orientation='vertical')
axip.set_title("W")



filepath = os.path.join(os.path.expanduser("~") + "/Dropbox/LMNphysio/model/model.pickle")
data = cPickle.load(open(filepath, 'rb'))
popcoh = data['popcoh']

gs_top2 = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outergs[0,1], wspace=0.9, hspace=0.9
    )

for i, st in enumerate(['adn', 'lmn']):
    for j, e in enumerate(['sws', 'opto']):

        subplot(gs_top2[i,j])
        simpleaxis(gca())
        H, xedges, yedges = np.histogram2d(popcoh[st]['wak'], popcoh[st][e], bins=np.linspace(-1, 1, 100))

        H[H>0] = 1
        
        imshow(H.T, origin='lower', extent=(-1, 1, -1, 1), cmap='binary')

        # scatter(popcoh[st]['wak'], popcoh[st][e], 0.5, color=colors[st])
        r, p = pearsonr(popcoh[st]['wak'], popcoh[st][e])
        m, b = np.polyfit(popcoh[st]['wak'], popcoh[st][e], 1)
        x = np.linspace(popcoh[st]['wak'].min(), popcoh[st]['wak'].max(),5)
        plot(x, x*m + b, color=COLOR, linewidth=1)

        xlim(-1, 1)
        ylim(-1, 1)
        xlabel("'Wake'")
        if e == 'sws':
            ylabel("'Sleep'")
        else:
            ylabel("'Opto.'")
        title(f" r={np.round(r, 2)}", y=0.9)

        if j == 0:
            text(1.5, 1.2, st.upper())


outergs.update(top=0.92, bottom=0.15, right=0.99, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/supp8.pdf",
    dpi=200,
    facecolor="white",
)

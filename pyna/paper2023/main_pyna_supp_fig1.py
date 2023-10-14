# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-10-14 17:25:33
import numpy as np
import pandas as pd
import pynapple as nap

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
import matplotlib.font_manager as font_manager

# matplotlib.style.use('seaborn-paper')
import matplotlib.image as mpimg

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv

import os
import sys

from scipy.ndimage import gaussian_filter

sys.path.append("../")
from functions import *

sys.path.append("../../python")
import neuroseries as nts


def figsize(scale):
    fig_width_pt = 483.69687  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2  # Aesthetic ratio (you could change this)
    # fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_width = 6
    fig_height = fig_width * golden_mean * 1.2  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
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

fontsize = 7

COLOR = (0.25, 0.25, 0.25)

rcParams["font.family"] = "sans-serif"
# rcParams['font.sans-serif'] = ['Arial']
rcParams["font.size"] = fontsize
rcParams["text.color"] = COLOR
rcParams["axes.labelcolor"] = COLOR
rcParams["axes.labelsize"] = fontsize
rcParams["axes.labelpad"] = 3
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
rcParams["axes.linewidth"] = 0.2
rcParams["axes.edgecolor"] = COLOR
rcParams["axes.axisbelow"] = True
rcParams["xtick.color"] = COLOR
rcParams["ytick.color"] = COLOR


colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}


# clrs = ['sandybrown', 'olive']
# clrs = ['#CACC90', '#8BA6A9']

###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"

data = cPickle.load(open(dropbox_path + "/DATA_SUPP_FIG_1_HMM_exemple.pickle", "rb"))

tcurves = data["tuning_curves"]
peaks = data["peaks"]
spikes = data["spikes"]
tokeep = data["tokeep"]
eps = data["eps"]
order = data["order"]

scores = data["scores"]


###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(2))

outergs = GridSpec(2, 1, figure=fig, height_ratios=[0.2, 0.3], hspace=0.4)

#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], width_ratios=[0.3, 0.3, 0.6], wspace=0.4
)

#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(
    4, 3, subplot_spec=gs1[0, 1])


for i, n in enumerate(order):    
    subplot(gs_tc[i//3, i%3], projection="polar")    
    tc = tcurves[tokeep[n]]
    fill_between(tc.index.values, np.zeros_like(tc), tc.values)
    plot(tc, linewidth=0.1, color=colors["lmn"], alpha=0.4)    
    


#########################
# RASTER PLOTS
#########################
ax = subplot(gs1[0,2])

mks = 1.1
alp = 1
medw = 0.08



#########################
# LOG
#########################
gs2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[1, 0], width_ratios=[0.3, 0.3, 0.6], wspace=0.4
)

ax = subplot(gs2[0,0])

for s in scores:
    plot(s, 'o-', markersize=1)



outergs.update(top=0.96, bottom=0.09, right=0.96, left=0.06)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2023/supp_fig1.pdf",
    dpi=200,
    facecolor="white",
)
# show()

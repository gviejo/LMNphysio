# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-04-01 13:47:26
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
import matplotlib.patches as patches

from scipy.stats import zscore

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

rcParams["font.family"] = 'DejaVu Sans Mono'
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
rcParams["axes.linewidth"] = 0.4
rcParams["axes.edgecolor"] = COLOR
rcParams["axes.axisbelow"] = True
rcParams["xtick.color"] = COLOR
rcParams["ytick.color"] = COLOR


colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}

cmap = plt.get_cmap("Set2")
# colors = {'adn':cmap(0), "lmn":cmap(1), "psb":cmap(2)}


# clrs = ['sandybrown', 'olive']
# clrs = ['#CACC90', '#8BA6A9']

###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"




###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, hspace = 0.3, height_ratios=[0.1, 0.1])


names = {'adn':"ADN", 'lmn':"LMN"}
epochs = {'wak':'Wake', 'sws':'Sleep'}

gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], width_ratios=[0.3, 0.3, 0.2], wspace=0.1
)


#####################################
# Histo
#####################################
gs_histo = gridspec.GridSpecFromSubplotSpec(
    1, 1, subplot_spec=gs_top[0, 0]#, height_ratios=[0.5, 0.2, 0.2] 
)


subplot(gs_histo[0,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN-PSB-opto.png")
imshow(img, aspect="equal")
xticks([])
yticks([])


#####################################
# Examples LMN IPSILATERAL
#####################################
st = 'adn'

gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, gs_top[0,1])

s = "A8000/A8066/A8066-240216B"
ex = nap.IntervalSet(4076.9, 4083.6)
path = os.path.join(data_directory, "OPTO", s)

spikes, position, eps = load_opto_data(path, st)
opto_ep = eps['opto_ep']

exex = nap.IntervalSet(ex.start[0] - 10, ex.end[0] + 10)
p = spikes.count(0.01, exex).smooth(0.04, size_factor=20)
d=np.array([p.loc[i] for i in spikes.order.sort_values().index]).T
p = nap.TsdFrame(t=p.t, d=d, time_support=p.time_support)
p = np.sqrt(p / p.max(0))
    
    
ax = subplot(gs2[0,0])
simpleaxis(gca())
for n in spikes.keys():
    # cl = hsv_to_rgb([spikes.peaks[n]/(2*np.pi), 1, 1])
    plot(spikes[n].restrict(ex).fillna(spikes.order[n]), '|', color=colors[st], markersize=2.1, mew=0.5)

[axvspan(s, e, alpha=0.1) for s, e in opto_ep.intersect(ex).values]
# ylim(-2, len(spikes)+2)
xticks([])
yticks([])


subplot(gs2[1,0])
noaxis(gca())
tmp = p.restrict(ex)
d = gaussian_filter(tmp.values, 1)
tmp2 = nap.TsdFrame(t=tmp.index.values, d=d)

pcolormesh(tmp2.index.values, 
        np.arange(0, tmp2.shape[1]),
        tmp2.values.T, cmap='GnBu', antialiased=True)
yticks([])

#####################################
# Firing rate
#####################################
gs_fr = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_top[0, 2]#, height_ratios=[0.5, 0.2, 0.2] 
)

subplot(gs_fr[0,0])


subplot(gs_fr[1,0])


##########################################
# ADN OPTO SLEEP
##########################################
data = cPickle.load(open(os.path.expanduser("~/Dropbox/LMNphysio/data/OPTO_SLEEP.pickle"), 'rb'))

allr = data['allr']
corr = data['corr']
change_fr = data['change_fr']




outergs.update(top=0.96, bottom=0.09, right=0.98, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/fig3.pdf",
    dpi=200,
    facecolor="white",
)
# show()

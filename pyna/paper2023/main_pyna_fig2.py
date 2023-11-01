# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-11-01 18:08:10
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
    fig_height = fig_width * golden_mean * 1  # height in inches
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

ex_path_psb = {
    "wake": "/DATA_FIG_PSB_WAKE_A8054-230719A.pickle",
    "sleep": "/DATA_FIG_PSB_SLEEP_A8054-230718A.pickle"
    }


###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

mks = 1.1
alp = 1
medw = 0.08


fig = figure(figsize=figsize(2))

outergs = GridSpec(3, 1, figure=fig, hspace=0.5)

#####################################
# PSB OPTO SLEEP
#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], 
    hspace=0.5, wspace = 0.3,
    width_ratios=[0.1, 0.5, 0.1]
)

# Histology
subplot(gs1[0, 0])
noaxis(gca())
img = mpimg.imread(dropbox_path+"/PSBopto.png")
imshow(img, aspect="equal", cmap="viridis")
xticks([])
yticks([])


# for e, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(-4,14), slice(-1,2)], [slice(-4,0), slice(-1,0)]):    

# PSB opto example wake
gs1_2 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs1[0, 1],
    hspace=0.4, wspace=0.5)

ep = 'wake'
sl = slice(-4, 14)
psbex = cPickle.load(open(dropbox_path + ex_path_psb[ep], "rb"))

for i, n, k in zip(range(2), psbex['tc'].columns, ['hd', 'nhd']):
    subplot(gs1_2[i,0], projection='polar')
    gca().spines["polar"].set_visible(False)
    fill_between(
        psbex['tc'][n].index.values,
        np.zeros_like(psbex['tc'][n].values),
        psbex['tc'][n].values,
        color = colors['psb']
        )
    xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['', '', '', ''])
    gca().tick_params(pad=1)
    yticks([])

    subplot(gs1_2[i,1])
    simpleaxis(gca())
    plot(psbex[k][np.arange(0, 30)].to_tsd(), '|', 
        color = colors['psb'],
        markersize=mks,
        markeredgewidth=medw,
        alpha=alp,
        )
    if i == 0:
        xticks([0, 10], ['', ''])
    if i == 1:
        xlabel("Time from stim (s)")

# PSB average firing rate
psbdata = cPickle.load(open(dropbox_path + "/OPTO_PSB.pickle", "rb"))
allmeta = psbdata['allmeta'][ep]
allfr = psbdata['allfr'][ep]
alltc = psbdata['alltc'][ep]

neurons = allmeta.index.values
hd = neurons[allmeta['SI']>0.5]
nhd = neurons[allmeta['SI']<=0.1]

for i, gr in enumerate([hd, nhd]):
    tc = centerTuningCurves(alltc[gr])
    tc = tc / tc.loc[0]
    subplot(gs1_2[i,2])
    simpleaxis(gca())
    plot(tc, color = colors['psb'], linewidth=0.1, alpha=0.4)
    plot(tc.mean(1), linewidth=0.5, color=colors['psb'])
    xticks([-np.pi, 0, np.pi], ['', '', ''])
    if i == 1:
        xticks([-np.pi, 0, np.pi], [-180, 0, 180])
        xlabel("Centered HD")


    tmp = allfr[gr].loc[sl]
    tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
    subplot(gs1_2[i,3])
    simpleaxis(gca())
    plot(tmp, color = colors['psb'], linewidth=0.1)
    # title(ep)
    ylim(0, 2.5)



# PSB opto vs control wake
subplot(gs1[0,2])
simpleaxis(gca())


#####################################
# LMN OPTO
#####################################
gs2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[1, 0], width_ratios=[0.2, 0.2, 0.1], hspace=0.5,
    wspace = 0.3
)

subplot(gs2[0, 0])
noaxis(gca())
img = mpimg.imread(dropbox_path+"/LMNopto.png")
imshow(img, aspect="equal", cmap="viridis")
xticks([])
yticks([])


gs22 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs2[0, 1],
    hspace=0.4, wspace=0.5)

lmndata = {
    "sleep":cPickle.load(open(dropbox_path + "/OPTO_LMN_sleep.pickle", "rb")),
    "wake":cPickle.load(open(dropbox_path + "/OPTO_LMN_wake.pickle", "rb"))
    }

for i, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(-4,14), slice(-1,2)], [slice(-4,0), slice(-1,0)]):
    allmeta = lmndata[ep]['allmeta']
    allfr = lmndata[ep]['allfr']
    order = allmeta.sort_values(by="SI").index.values
    tmp = allfr[order].loc[sl]
    tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
    subplot(gs22[0,i])
    simpleaxis(gca())
    plot(tmp, color = colors['lmn'], linewidth=0.1)
    ylim(0, 2.5)
    title(ep)

    subplot(gs22[1,i])
    # tmp = tmp - tmp.loc[msl].mean(0)
    # tmp = tmp / tmp.std(0)    
    imshow(tmp.values.T)
    

#####################################
# LMN CORRELATION
#####################################
gs3 = gridspec.GridSpecFromSubplotSpec(
    1, 5, subplot_spec=outergs[2, 0], wspace = 1, 
    width_ratios=[0.01,0.3, 0.3, 0.2, 0.4]
    )

allr = lmndata['sleep']['allr']
corr = lmndata['sleep']['corr']
epochs = ['sws', 'opto']
for i, e in enumerate(epochs):
    subplot(gs3[0,i+1])
    simpleaxis(gca())
    plot(allr['wak'], allr[e], 'o', color = 'red', alpha = 0.5, markersize=1)
    m, b = np.polyfit(allr['wak'].values, allr[e].values, 1)
    x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
    plot(x, x*m + b)
    xlabel('wak')
    ylabel(e)
    xlim(allr['wak'].min(), allr['wak'].max())
    ylim(allr.iloc[:,1:].min().min(), allr.iloc[:,1:].max().max())
    r, p = scipy.stats.pearsonr(allr['wak'], allr[e])
    title('r = '+str(np.round(r, 3)))

subplot(gs3[0,3])
simpleaxis(gca())
for i, e in enumerate(corr.columns):
    plot(np.random.randn(len(corr))*0.1+np.ones(len(corr))*i, corr[e], 'o', markersize=5)
ylim(0, 1)
xticks(np.arange(corr.shape[1]), corr.columns)




outergs.update(top=0.95, bottom=0.1, right=0.95, left=0.06)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2023/fig2.pdf",
    dpi=200,
    facecolor="white",
)
# show()

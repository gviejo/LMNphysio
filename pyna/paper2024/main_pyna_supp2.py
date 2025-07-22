# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-07-19 15:01:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-22 17:57:28
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
    fig_height = fig_width * golden_mean * 2.0 # height in inches
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

outergs = GridSpec(3, 1, hspace = 0.3, height_ratios=[0.2, 0.6, 0.3])


#########################
# TUNING CURVES SELECTION
#########################

gs1 = gridspec.GridSpecFromSubplotSpec(1,4, outergs[0,0], 
    wspace = 0.6, width_ratios=[-0.1, 0.5, 0.5, 0.25]
    )

for i, st in enumerate(['psb']):
    gs1_1 = gridspec.GridSpecFromSubplotSpec(2,2, gs1[0,i+1], wspace=0.5, hspace=0.5        
        )

    idx = allinfo[allinfo['location'] == st].index.values

    tokeep = allinfo.loc[idx][(allinfo.loc[idx]['rate'] > 1.0) & (allinfo.loc[idx]['SI'] > 0.1) & (allinfo.loc[idx]['tokeep']==1)].index.values


    # Cloud
    ax = subplot(gs1_1[:,0])
    simpleaxis(ax)

    loglog(allinfo.loc[idx, 'rate'], allinfo.loc[idx, 'SI'], '.', markersize=1, color=COLOR)
    loglog(allinfo.loc[tokeep, 'rate'], allinfo.loc[tokeep, 'SI'], 'o', color=colors[st], markersize=1)

    xlabel("Rate (Hz)")
    ylabel("Mutual Information (bits/spk)", labelpad=5)

    axvline(1.0, linewidth=0.5, color=COLOR)
    axhline(0.1, linewidth=0.5, color=COLOR)

    tc = centerTuningCurves_with_peak(alltc[tokeep])
    tc = tc/tc.max()

    # Tuning curves
    subplot(gs1_1[0,1])
    simpleaxis(gca())
    m = tc.mean(1)
    s = tc.std(1)
    plot(tc.mean(1), linewidth=1, color=colors[st])
    fill_between(m.index.values, m.values-s.values, m.values+s.values, color=colors[st], alpha=0.2)
    xticks([-np.pi, np.pi], [-180, 180])
    xlim(-np.pi, np.pi)

    ax = subplot(gs1_1[1,1])
    tmp = tc.values.T
    im = imshow(tmp, aspect='auto')

    xticks([0, tmp.shape[1]], [-180, 180])
    yticks([0], [tmp.shape[0]])
    xlabel("Angle (deg.)", labelpad=-1)

    axip = gca().inset_axes([0, -0.65, 0.5, 0.08])
    noaxis(axip)
    cbar = colorbar(im, cax=axip, orientation='horizontal')
    axip.text(1.05, 0.5, "Rate", transform=axip.transAxes,
          va='center', ha='left')
    # axip.set_title("Rate")
    # axip.set_yticks([0, 3])


#############################################################
# OPTO PSB - PSB
#############################################################
gs1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[1, 0], 
    hspace=0.5, wspace = 0.3,
    width_ratios=[0.1, 0.4, 0.1]
)

# Histology
subplot(gs1[0, 0])
noaxis(gca())
# img = mpimg.imread(dropbox_path+"/PSBopto.png")
# imshow(img, aspect="equal", cmap="viridis")
# xticks([])
# yticks([])



gs1_2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs1[0, 1],
    hspace=0.5, wspace=0.6)


psbdata = cPickle.load(open(dropbox_path + "/OPTO_PSB.pickle", "rb"))



# PSB all tuning curves separated
tc_hd = []
tc_nhd = []

fr = {'hd':{}, 'nhd':{}}

groups = {}
for ep in ['wake', 'sleep']:
    allmeta = psbdata['allmeta'][ep]
    allfr = psbdata['allfr'][ep]
    alltc = psbdata['alltc'][ep]
    neurons = allmeta.index.values
    hd = neurons[allmeta['SI']>0.5]
    nhd = neurons[allmeta['SI']<0.5]    
    if ep == 'wake':
        tmp = allfr[nhd].loc[0:10].mean() - allfr[nhd].loc[-4:0].mean()         
        nhd = tmp[tmp>0].index.values
    if ep == 'sleep':
        tmp = allfr[nhd].loc[0:1].mean() - allfr[nhd].loc[-1:0].mean()         
        nhd = tmp[tmp>0].index.values

    tc_hd.append(alltc[hd])
    tc_nhd.append(alltc[nhd])
    groups[ep] = {'hd':hd, 'nhd':nhd}
    fr['hd'][ep] = allfr[hd]
    fr['nhd'][ep] = allfr[nhd]


tc_hd = pd.concat(tc_hd, 1)
tc_nhd = pd.concat(tc_nhd, 1)

titles2 = ['PoSub HD', 'PoSub non-HD']

for i, tc in enumerate([tc_hd, tc_nhd]):
    tc = centerTuningCurves2(tc)
    tc = tc / tc.loc[0]
    subplot(gs1_2[i,0])
    simpleaxis(gca())
    plot(tc, color = clrs[i], linewidth=0.1, alpha=0.5)
    plot(tc.mean(1), linewidth=1, color=clrs[i])
    xticks([-np.pi, 0, np.pi], ['', '', ''])
    yticks([0, 1], ['0', '1'])
    title(titles2[i], pad = 2)
    if i == 1:
        xticks([-np.pi, 0, np.pi], [-180, 0, 180])
        xlabel("Centered HD", labelpad=1)
        ylabel("Firing rate (norm.)", y = 1)

titles = ['Wake', 'nREM sleep']


for e, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(-4,14), slice(-1,2)], [slice(-4,0), slice(-1,0)]):    
    allfr = psbdata['allfr'][ep]    
    for i, gr in enumerate(['hd', 'nhd']):
        tmp = allfr[groups[ep][gr]].loc[sl]
        # tmp = tmp.rolling(window=100, win_type='gaussian', center=True, min_periods=1, axis = 0).mean(std=1)
        subplot(gs1_2[i,e+1])
        simpleaxis(gca())
        plot(tmp, color = clrs[i], linewidth=0.1, alpha=0.5)
        plot(tmp.mean(1), color = clrs[i], linewidth=1.0)
        if ep == 'sleep':
            axvspan(0, 1, color = 'lightcoral', alpha=0.2, ec = None)
            xticks([0, 1])
        if ep == 'wake':
            axvspan(0, 10, color = 'lightcoral', alpha=0.2, ec = None)
            xticks([0, 10])
        if i == 0:
            title(titles[e])        
        # ylim(0, 2.5)
        yticks([0, 2], ['0', '2'])
        if i == 1:
            ylabel("Firing rate (norm.)", y=1.5)
            xlabel("Time (s)", labelpad=1)

titles = ['Wake', 'nREM']

# PSB opto vs control wake
gs1_3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[0, 2],
    hspace=0.5, wspace=0.6)
for i, gr in enumerate(['hd', 'nhd']):
    xs = [[0, 1], [2,3]]

    for e, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(0,10), slice(0,1)], [slice(-3,-1), slice(-1,0)]):

        subplot(gs1_3[i, e])
        simpleaxis(gca())


        allfr = psbdata['allfr'][ep]
            
        tmp = allfr[groups[ep][gr]]

        y = tmp.loc[sl].mean().values-tmp.loc[msl].mean().values

        hist(y, edgecolor=COLOR, facecolor='white', bins = np.linspace(-1, 1, 8))

        if i == 0:
            xlim(-1, 0)            
            title(titles[e])
                    
        if i == 1 and e == 0:
            ylabel("Count", y = 1.05, labelpad = 10)
            xlabel("% mod", x = 1)
        # if i == 1:
        xlim(-1, 1)
        # xlim(-1, 1)
        # xticks([-0.5, 0, 0.5], [-50, 0, 50])


        # corr = np.vstack((tmp2, tmp1)).T

        # plot(xs[e], corr.T, color = clrs[i], linewidth=0.1)
        # plot(xs[e], corr.T.mean(1), color = clrs[i], linewidth=1)

    # yticks([0, 1, 2])
    # xticks([])
    # if i == 1:
    #     ylabel("Rate mod. (norm.)", y = 1.5)#, labelpad = 8)
    #     xticks([0.5, 2.5], ["Wake", "nREM"])







outergs.update(top=0.97, bottom=0.04, right=0.97, left=0.08)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/supp2.pdf",
    dpi=200,
    facecolor="white",
)
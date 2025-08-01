# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-07-19 15:01:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-08-01 16:48:23
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
    fig_height = fig_width * golden_mean * 1.5 # height in inches
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
data = cPickle.load(open(os.path.join(dropbox_path, 'All_TC.pickle'), 'rb'))

alltc = data['alltc']
allinfo = data['allinfo']



fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, hspace = 0.3, height_ratios=[0.1, 0.5])


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
    ylabel("HD Info. (bits/spk)", labelpad=5)

    axvline(1.0, linewidth=0.5, color=COLOR)
    axhline(0.1, linewidth=0.5, color=COLOR)

    tc = centerTuningCurves_with_peak(alltc[tokeep])
    tc = tc/tc.max()    

    # Tuning curves HD
    subplot(gs1_1[0,1])
    simpleaxis(gca())
    m = tc.mean(1)
    s = tc.std(1)
    plot(tc.mean(1), linewidth=1, color=colors[st])
    fill_between(m.index.values, m.values-s.values, m.values+s.values, color=colors[st], alpha=0.2)
    xticks([-np.pi, np.pi], [-180, 180])
    xlim(-np.pi, np.pi)
    title("PSB HD")

    ax = subplot(gs1_1[1,1])
    tmp = tc.values.T
    im = imshow(tmp, aspect='auto')

    xticks([0, tmp.shape[1]], [-180, 180])
    yticks([0], [tmp.shape[0]])
    xlabel("Angle (deg.)", labelpad=-1)

    axip = gca().inset_axes([0, -0.8, 0.5, 0.08])
    noaxis(axip)
    cbar = colorbar(im, cax=axip, orientation='horizontal')
    axip.text(1.05, 0.5, "Rate", transform=axip.transAxes,
          va='center', ha='left')
    # axip.set_title("Rate")
    # axip.set_yticks([0, 3])



#############################################################
# OPTO PSB - PSB
#############################################################



gs2 = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outergs[1, 0], hspace=0.4
)


#################################################
# RASTER EXAMPLE
#################################################


filepath = os.path.join(os.path.expanduser("~"), 'Dropbox/LMNphysio/data/DATA_FIG_PSB_SLEEP_A8054-230718A.pickle')

data = cPickle.load(open(filepath, 'rb'))

hd = data['hd']
nhd = data['nhd']
tc = data['tc']
mod = data['mod']
peth = data['peth']
# mod = mod.sort_values()
neurons = [
    mod[hd].dropna().sort_values().index.values[0:4],
    mod[nhd].dropna().sort_values().index.values[-5:-1][::-1],
]

colors2 = [colors['psb'], "grey"]


for i, idx in enumerate(neurons):
    gs2_2 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs2[0,i],
        hspace=0.4,wspace = 0.3        
    )
    inds = np.indices((1,4))
    zipind = np.stack((inds[0].flat, inds[1].flat)).T

    for j, n in enumerate(idx):
        gs2_3 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs2_2[zipind[j,0], zipind[j,1]],
            hspace=0.1, wspace = 0.3,
            height_ratios=[0.2, 0.4]
        )

        subplot(gs2_3[1,0])
        simpleaxis(gca())
        plot(peth[n].to_tsd(), '|', color=colors2[i], markersize=0.5, markeredgewidth=0.1)
        xticks([0, 1])
        yticks([])
        xlabel("Lag (s)")
        ylim(0, len(peth[n])+16)
        gca().spines['left'].set_bounds(0, len(peth[n]))

        if j == 0:
            ylabel("Stim.")
            yticks([len(peth[n])])

        rect = patches.Rectangle((0, len(peth[n])+1), width=1.0, height=15,
            linewidth=0, facecolor="lightcoral")
        gca().add_patch(rect)


        subplot(gs2_3[0,0], projection='polar')
        plot(tc[n], color=colors2[i])
        xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['', '', '', ''])
        yticks([])

#################################################
# OPTO
#################################################
psbdata = cPickle.load(open(dropbox_path + "/OPTO_PSB.pickle", "rb"))


gs2_2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs2[1, :],
    hspace=0.5, wspace = 0.3,
    width_ratios=[0.5, 0.2]
)


gs1_2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs2_2[0, 0],
    hspace=0.5, wspace=0.6)

# PSB all tuning curves separated
tc_hd = []
tc_nhd = []

fr = {}
groups = {}

for gr in ["opto", "ctrl"]:
    fr[gr] = {}
    groups[gr] = {}
    for ep in ['wake', 'sleep']:
        fr[gr][ep] = {}
        allmeta = psbdata['allmeta'][gr][ep]
        allfr = psbdata['allfr'][gr][ep]
        alltc = psbdata['alltc'][gr][ep]
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
        groups[gr][ep] = {'hd':hd, 'nhd':nhd}
        fr[gr][ep] = {"hd":allfr[hd], "nhd":allfr[nhd]}        


tc_hd = pd.concat(tc_hd, axis=1)
tc_nhd = pd.concat(tc_nhd, axis=1)

titles2 = ['PoSub HD', 'PoSub non-HD']

for i, tc in enumerate([tc_hd, tc_nhd]):
    tc = centerTuningCurves_with_peak(tc)
    tc = tc / tc.loc[0]
    subplot(gs1_2[i,0])
    simpleaxis(gca())
    plot(tc, color = colors2[i], linewidth=0.1, alpha=0.5)
    plot(tc.mean(1), linewidth=1, color=COLOR)
    xticks([-np.pi, 0, np.pi], ['', '', ''])
    yticks([0, 1], ['0', '1'])
    title(titles2[i], pad = 2)
    if i == 1:
        xticks([-np.pi, 0, np.pi], [-180, 0, 180])
        xlabel("Centered HD", labelpad=1)
        ylabel("Firing rate (norm.)", y = 1)

titles = ['Wake', 'nREM sleep']

gr = "opto"

for e, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(-4,14), slice(-1,2)], [slice(-4,0), slice(-1,0)]):    
    allfr = psbdata['allfr'][gr][ep]
    for i, grn in enumerate(['hd', 'nhd']):
        tmp = allfr[groups[gr][ep][grn]].loc[sl]
        # tmp = tmp.rolling(window=100, win_type='gaussian', center=True, min_periods=1, axis = 0).mean(std=1)
        subplot(gs1_2[i,e+1])
        simpleaxis(gca())
        tmp = tmp.apply(lambda col: gaussian_filter1d(col.values, sigma=1), axis=0)
        plot(tmp, color = colors2[i], linewidth=0.1, alpha=0.5)
        plot(tmp.mean(1), color = COLOR, linewidth=1.0)
        if ep == 'sleep':
            axvspan(0, 1, color = 'lightcoral', alpha=0.2, ec = None)
            xticks([0, 1])
        if ep == 'wake':
            axvspan(0, 10, color = 'lightcoral', alpha=0.2, ec = None)
            xticks([0, 10])
        if i == 0:
            title(titles[e])        
        ylim(0, 3)
        yticks([0, 3], ['0', '2'])
        if i == 1:
            ylabel("Firing rate (norm.)", y=1.5)
            xlabel("Time (s)", labelpad=1)

titles = ['Wake', 'nREM']

# PSB opto vs control wake
gs1_3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs2_2[0, 1],
    hspace=0.3, wspace=0.6)


colors2 = ["#DC143C", "#FF7F50"]

for i, grn in enumerate(['hd', 'nhd']):
    xs = [[0, 1], [2,3]]

    for e, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(1,10), slice(0.1,1)], [slice(-5,-1), slice(-1,-0.1)]):

        subplot(gs1_3[i, e])
        simpleaxis(gca())
        gca().spines['bottom'].set_bounds(0, 1)
        ys = []

        for j, gr in enumerate(['opto', 'ctrl']):

            allfr = psbdata['allfr'][gr][ep]
            
            tmp = allfr[groups[gr][ep][grn]]

            y = (tmp.loc[sl].mean().values-tmp.loc[msl].mean().values)/(tmp.loc[sl].mean().values+tmp.loc[msl].mean().values)

            plot(np.ones_like(y)*j+np.random.randn(len(y))*0.1, y, 'o', color=colors2[j], markersize=1)
            
            plot([j-0.2, j+0.2], [np.mean(y), np.mean(y)], linewidth=1, color=COLOR)            

            ys.append(y)


        zw, p = scipy.stats.mannwhitneyu(ys[0], ys[1])

        signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])

        xl, xr = 1.5, 1.6
        m = [ys[0].mean(), ys[1].mean()]
        plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
        plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
        plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
        text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")

        print(grn, ep, zw, p, len(ys[0]), len(ys[1]))

        xlim(-0.5, 2)
        ylim(-1, 1)
        yticks([-1, 0, 1])
        if e == 0:
            ylabel("Mod.")
        if i == 0:
            title(titles[e])
            xticks([0,1], ["", ""])
        else:
            xticks([0, 1], ["Chrim.", 'TdTom.'], rotation=45)

        # hist(y, edgecolor=COLOR, facecolor='white', bins = np.linspace(-1, 1, 16))

        # if i == 0:
        #     xlim(-1, 0)            
        #     title(titles[e])
                    
        # if i == 1 and e == 0:
        #     ylabel("Count", y = 1.05, labelpad = 10)
        #     xlabel("% mod", x = 1)
        # # if i == 1:
        # xlim(-1, 1)

        # zw, p = scipy.stats.wilcoxon(y)

        # print(f"Rate mod {gr} {grn} {ep}", "Wilcoxon", zw, p, "n=", len(y))

        # signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
        # text(0.6, 0.8, s=map_significance[signi], va="center", ha="left", transform=gca().transAxes)







outergs.update(top=0.97, bottom=0.06, right=0.97, left=0.08)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/supp3.pdf",
    dpi=200,
    facecolor="white",
)
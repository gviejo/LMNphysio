# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-05-14 17:06:58
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
cycle = rcParams['axes.prop_cycle'].by_key()['color']

rcParams["font.family"] = 'Liberation Sans'
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
rcParams['xtick.major.pad'] = 0.5
rcParams['ytick.major.pad'] = 0.5
rcParams['xtick.minor.pad'] = 0.5
rcParams['ytick.minor.pad'] = 0.5

colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}

cmap = plt.get_cmap("Set2")
# colors = {'adn':cmap(0), "lmn":cmap(1), "psb":cmap(2)}


# clrs = ['sandybrown', 'olive']
# clrs = ['#CACC90', '#8BA6A9']

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

outergs = GridSpec(2, 1, hspace = 0.5, height_ratios=[0.1, 0.1])


names = {'adn':"ADN", 'lmn':"LMN"}
epochs = {'wak':'Wake', 'sws':'Sleep'}


#####################################
# ADN IPSILATERAL
#####################################


gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[0, 0], width_ratios=[0.05, 0.9], wspace=0.3
)


#####################################
# HISTO
#####################################
gs_histo = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_top[0, 0], hspace=0.0
)


subplot(gs_histo[0,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN-PSB-opto.png")
imshow(img, aspect="equal")
xticks([])
yticks([])


subplot(gs_histo[1,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/A8066_S7_3_2xMerged.png")
imshow(img, aspect="equal")
xticks([])
yticks([])


subplot(gs_histo[2,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/A8066_S7_3_4xMerged.png")
imshow(img, aspect="equal")
xticks([])
yticks([])

#####################################
# Examples ADN IPSILATERAL
#####################################
st = 'adn'

gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, gs_top[0,1], hspace=0.5, wspace=0.9)


exs = [
    ("B3700/B3704/B3704-240609A", nap.IntervalSet(5130, 5232), "Wakefulness"),
    ("B3700/B3704/B3704-240608A", nap.IntervalSet(4110.6, 4117.5), "non-REM sleep")
]

for i, (s, ex, name) in enumerate(exs):

    path = os.path.join(data_directory, "OPTO", s)

    spikes, position, wake_ep, opto_ep, sws_ep = load_opto_data(path, st)

    # Decoding 

    
    gs2_ex = gridspec.GridSpecFromSubplotSpec(2, 1, gs2[i,0], hspace = 0.3)

    subplot(gs2_ex[0,0])
    simpleaxis(gca())    
    gca().spines['bottom'].set_visible(False)
    ms = [0.7, 0.9]
    plot(spikes.to_tsd("order").restrict(ex), '|', color=colors[st], markersize=ms[i], mew=0.25)

    [axvspan(s, e, alpha=0.2, color="lightsalmon") for s, e in opto_ep.intersect(ex).values]
    # ylim(-2, len(spikes)+2)
    xlim(ex.start[0], ex.end[0])
    xticks([])
    yticks([0, len(spikes)-1], [1, len(spikes)])
    gca().spines['left'].set_bounds(0, len(spikes)-1)
    title(name)
    ylabel("ADN")


    exex = nap.IntervalSet(ex.start[0] - 10, ex.end[0] + 10)
    p = spikes.count(0.01, exex).smooth(0.04, size_factor=20)
    d=np.array([p.loc[i] for i in spikes.index[np.argsort(spikes.order)]]).T
    p = nap.TsdFrame(t=p.t, d=d, time_support=p.time_support)
    p = np.sqrt(p / p.max(0))
    # p = 100*(p / p.max(0))


    subplot(gs2_ex[1,0])
    simpleaxis(gca())
    tmp = p.restrict(ex)
    d = gaussian_filter(tmp.values, 2)
    tmp2 = nap.TsdFrame(t=tmp.index.values, d=d)

    im = pcolormesh(tmp2.index.values, 
            np.linspace(0, 2*np.pi, tmp2.shape[1]),
            tmp2.values.T, cmap='GnBu', antialiased=True)

    x = np.linspace(0, 2*np.pi, tmp2.shape[1])
    yticks([0, 2*np.pi], [1, len(spikes)])
    vls = opto_ep.intersect(ex).values[0] 
    gca().spines['bottom'].set_bounds(vls[0], vls[1])        
    xticks(vls, ['', ''])
    if i == 0: xlabel("10 s", labelpad=-1)
    if i == 1: xlabel("1 s", labelpad=-1)

    if name == "Wakefulness":
        tmp = position['ry'].restrict(ex)
        iset=np.abs(np.gradient(tmp)).threshold(1.0, method='below').time_support
        for s, e in iset.values:
            plot(tmp.get(s, e), linewidth=0.5, color=COLOR)

    # Colorbar
    axip = gca().inset_axes([1.05, 0.0, 0.05, 0.75])
    noaxis(axip)
    cbar = colorbar(im, cax=axip)
    axip.set_title("r", y=0.75)
    axip.set_yticks([0.25, 0.75])



##########################################
# ADN OPTO IPSILATERAL
##########################################

# gs_corr = gridspec.GridSpecFromSubplotSpec(
#     2, 2, subplot_spec=gs_bottom[0, 2], hspace=1.0#, height_ratios=[0.5, 0.2, 0.2] 
# )

allorders = {"OPTO_SLEEP": [('adn', 'opto', 'ipsi', 'opto'), 
                        ('adn', 'opto', 'ipsi', 'sws'),],
            "OPTO_WAKE": [('adn', 'opto', 'ipsi', 'opto'), 
                        ('adn', 'opto', 'ipsi', 'pre'),]}


ranges = {
    "OPTO_SLEEP":(-0.9,0,1,1.9),
    "OPTO_WAKE":(-9,0,10,19)
    }



titles = ['Wakefulness', 'non-REM sleep']

for i, f in enumerate(['OPTO_WAKE', 'OPTO_SLEEP']):

    data = cPickle.load(open(os.path.expanduser(f"~/Dropbox/LMNphysio/data/{f}.pickle"), 'rb'))

    allr = data['allr']
    corr = data['corr']
    change_fr = data['change_fr']
    allfr = data['allfr']
    baseline = data['baseline']

    orders = allorders[f]

    ####################
    # FIRING rate change
    ####################
    gs_corr2 = gridspec.GridSpecFromSubplotSpec(1,2, gs2[i,1], width_ratios=[0.2, 0.1])

    subplot(gs_corr2[0,0])
    simpleaxis(gca())

    keys = orders[0]
    tmp = allfr[keys[0]][keys[1]][keys[2]]
    tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1.5, mode='constant'))
    tmp = tmp.loc[ranges[f][0]:ranges[f][-1]]
    m = tmp.mean(1)
    s = tmp.std(1)
    
    # plot(tmp, color = 'grey', alpha=0.2)
    plot(tmp.mean(1), color = COLOR)
    fill_between(m.index.values, m.values-s.values, m.values+s.values, color=COLOR, alpha=0.2, ec=None)
    axvspan(ranges[f][1], ranges[f][2], alpha=0.2, color='lightsalmon', edgecolor=None)    
    if i == 1: xlabel("Stim. time (s)", labelpad=1)
    xlim(ranges[f][0], ranges[f][-1])
    # ylim(0.0, 4.0)
    title(titles[i])
    xticks([ranges[f][1], ranges[f][2]])
    ylim(0, 2)
    if i == 0:
        ylabel("Rate\n(norm.)")

    ################
    # PEARSON Correlation
    ################
    gs_corr3 = gridspec.GridSpecFromSubplotSpec(2,1, gs2[i,2], hspace=0.9, wspace=0.2)
    
    subplot(gs_corr3[0,0])
    simpleaxis(gca())
    title("Sessions")

    corr3 = []
    base3 = []    
    for j, keys in enumerate(orders):
        st, gr, sd, k = keys

        corr2 = corr[st][gr][sd]
        corr2 = corr2[corr2['n']>4][k]
        idx = corr2.index.values
        corr2 = corr2.values.astype(float)
        corr3.append(corr2)
        base3.append(baseline[st][gr][sd][k].loc[idx].values.astype(float))

        plot(np.random.randn(len(corr2))*0.05+np.ones(len(corr2))*(j+1), corr2, '.', markersize=1)
    
    
    xlim(0.5, 3)
    gca().spines['bottom'].set_bounds(1, 2)
    ymin = corr3[0].min()
    ylim(ymin-0.1, 1.1)
    gca().spines['left'].set_bounds(ymin-0.1, 1.1)

    ylabel("Pearson r")
    xticks([1, 2], ['Chrimson', 'Ctrl'], fontsize=fontsize-1)

    # if i == 1: 
    #     yticks([])
    #     gca().spines["left"].set_visible(False)

    ############
    subplot(gs_corr3[1,0])
    simpleaxis(gca())
    xlim(0.5, 3)
    ylim(-0.5, 1)
    gca().spines['bottom'].set_bounds(1, 2)
    xlabel("minus baseline", labelpad=1)
    plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
    plot([2.2], [0], 'o', color=COLOR, markersize=0.5)


    tmp = [c-b for c, b in zip(corr3, base3)]
    vp = violinplot(tmp, showmeans=False, 
        showextrema=False, vert=True, side='high'
        )    
    for k, p in enumerate(vp['bodies']): p.set_color(cycle[k])

    m = [c.mean() for c in corr3]
    plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

    xticks([1,2],['',''])
    ylabel(r"Mean$\Delta$")
    # if i == 1: 
    #     yticks([])
    #     gca().spines["left"].set_visible(False)

    # COmputing tests    
    
    for i in range(2):
        zw, p = scipy.stats.mannwhitneyu(corr3[i], base3[i], alternative='greater')
        signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
        text(i+0.9, m[i]-0.07, s=map_significance[signi], va="center", ha="right")

    xl, xr = 2.5, 2.6
    plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
    plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
    plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
    zw, p = scipy.stats.mannwhitneyu(tmp[0], tmp[1])
    signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")



#####################################
# ADN BILATERAL
#####################################


gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[1, 0], width_ratios=[0.05, 0.9], wspace=0.3
)


#####################################
# Histo
#####################################
gs_histo = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_bottom[0, 0], hspace=0.0
)


subplot(gs_histo[0,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN-PSB-opto.png")
imshow(img, aspect="equal")
xticks([])
yticks([])


subplot(gs_histo[1,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/A8066_S7_3_2xMerged.png")
imshow(img, aspect="equal")
xticks([])
yticks([])


subplot(gs_histo[2,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/A8066_S7_3_4xMerged.png")
imshow(img, aspect="equal")
xticks([])
yticks([])

#####################################
# Examples ADN BILATERAL
#####################################
st = 'adn'

gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, gs_bottom[0,1], hspace=0.5, wspace=0.9)


exs = [
    ("B3700/B3704/B3704-240609A", nap.IntervalSet(5130, 5232), "Wakefulness"),
    ("B3700/B3704/B3704-240608A", nap.IntervalSet(4110.6, 4117.5), "non-REM sleep")
]

for i, (s, ex, name) in enumerate(exs):

    path = os.path.join(data_directory, "OPTO", s)

    spikes, position, wake_ep, opto_ep, sws_ep = load_opto_data(path, st)

    # Decoding 

    
    gs2_ex = gridspec.GridSpecFromSubplotSpec(2, 1, gs2[i,0], hspace = 0.3)

    subplot(gs2_ex[0,0])
    simpleaxis(gca())    
    gca().spines['bottom'].set_visible(False)
    ms = [0.7, 0.9]
    plot(spikes.to_tsd("order").restrict(ex), '|', color=colors[st], markersize=ms[i], mew=0.25)

    [axvspan(s, e, alpha=0.2, color="lightsalmon") for s, e in opto_ep.intersect(ex).values]
    # ylim(-2, len(spikes)+2)
    xlim(ex.start[0], ex.end[0])
    xticks([])
    yticks([0, len(spikes)-1], [1, len(spikes)])
    gca().spines['left'].set_bounds(0, len(spikes)-1)
    title(name)
    ylabel("ADN")


    exex = nap.IntervalSet(ex.start[0] - 10, ex.end[0] + 10)
    p = spikes.count(0.01, exex).smooth(0.04, size_factor=20)
    d=np.array([p.loc[i] for i in spikes.index[np.argsort(spikes.order)]]).T
    p = nap.TsdFrame(t=p.t, d=d, time_support=p.time_support)
    p = np.sqrt(p / p.max(0))
    # p = 100*(p / p.max(0))


    subplot(gs2_ex[1,0])
    simpleaxis(gca())
    tmp = p.restrict(ex)
    d = gaussian_filter(tmp.values, 2)
    tmp2 = nap.TsdFrame(t=tmp.index.values, d=d)

    im = pcolormesh(tmp2.index.values, 
            np.linspace(0, 2*np.pi, tmp2.shape[1]),
            tmp2.values.T, cmap='GnBu', antialiased=True)

    x = np.linspace(0, 2*np.pi, tmp2.shape[1])
    yticks([0, 2*np.pi], [1, len(spikes)])
    vls = opto_ep.intersect(ex).values[0] 
    gca().spines['bottom'].set_bounds(vls[0], vls[1])        
    xticks(vls, ['', ''])
    if i == 0: xlabel("10 s", labelpad=-1)
    if i == 1: xlabel("1 s", labelpad=-1)

    if name == "Wakefulness":
        tmp = position['ry'].restrict(ex)
        iset=np.abs(np.gradient(tmp)).threshold(1.0, method='below').time_support
        for s, e in iset.values:
            plot(tmp.get(s, e), linewidth=0.5, color=COLOR)

    # Colorbar
    axip = gca().inset_axes([1.05, 0.0, 0.05, 0.75])
    noaxis(axip)
    cbar = colorbar(im, cax=axip)
    axip.set_title("r", y=0.75)
    axip.set_yticks([0.25, 0.75])



##########################################
# ADN OPTO BILATERAL
##########################################

# gs_corr = gridspec.GridSpecFromSubplotSpec(
#     2, 2, subplot_spec=gs_bottom[0, 2], hspace=1.0#, height_ratios=[0.5, 0.2, 0.2] 
# )


allorders = {"OPTO_SLEEP": [('adn', 'opto', 'ipsi', 'opto'), 
                        ('adn', 'opto', 'ipsi', 'sws'),],
            "OPTO_WAKE": [('adn', 'opto', 'ipsi', 'opto'), 
                        ('adn', 'opto', 'ipsi', 'pre'),]}

ranges = {
    "OPTO_SLEEP":(-0.9,0,1,1.9),
    "OPTO_WAKE":(-9,0,10,19)
    }



titles = ['Wakefulness', 'non-REM sleep']

for i, f in enumerate(['OPTO_WAKE', 'OPTO_SLEEP']):

    data = cPickle.load(open(os.path.expanduser(f"~/Dropbox/LMNphysio/data/{f}.pickle"), 'rb'))

    allr = data['allr']
    corr = data['corr']
    change_fr = data['change_fr']
    allfr = data['allfr']
    baseline = data['baseline']

    orders = allorders[f]

    ####################
    # FIRING rate change
    ####################
    gs_corr2 = gridspec.GridSpecFromSubplotSpec(1,2, gs2[i,1], width_ratios=[0.2, 0.1])

    subplot(gs_corr2[0,0])
    simpleaxis(gca())

    keys = orders[0]
    tmp = allfr[keys[0]][keys[1]][keys[2]]
    tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1.5, mode='constant'))
    tmp = tmp.loc[ranges[f][0]:ranges[f][-1]]
    m = tmp.mean(1)
    s = tmp.std(1)
    
    # plot(tmp, color = 'grey', alpha=0.2)
    plot(tmp.mean(1), color = COLOR)
    fill_between(m.index.values, m.values-s.values, m.values+s.values, color=COLOR, alpha=0.2, ec=None)
    axvspan(ranges[f][1], ranges[f][2], alpha=0.2, color='lightsalmon', edgecolor=None)    
    if i == 1: xlabel("Stim. time (s)", labelpad=1)
    xlim(ranges[f][0], ranges[f][-1])
    # ylim(0.0, 4.0)
    title(titles[i])
    xticks([ranges[f][1], ranges[f][2]])
    ylim(0, 2)
    if i == 0:
        ylabel("Rate\n(norm.)")

    ################
    # PEARSON Correlation
    ################
    gs_corr3 = gridspec.GridSpecFromSubplotSpec(2,1, gs2[i,2], hspace=0.9, wspace=0.2)
    
    subplot(gs_corr3[0,0])
    simpleaxis(gca())
    title("Sessions")

    corr3 = []
    base3 = []    
    for j, keys in enumerate(orders):
        st, gr, sd, k = keys

        corr2 = corr[st][gr][sd]
        corr2 = corr2[corr2['n']>4][k]
        idx = corr2.index.values
        corr2 = corr2.values.astype(float)
        corr3.append(corr2)
        base3.append(baseline[st][gr][sd][k].loc[idx].values.astype(float))

        plot(np.random.randn(len(corr2))*0.05+np.ones(len(corr2))*(j+1), corr2, '.', markersize=1)
    
    
    xlim(0.5, 3)
    gca().spines['bottom'].set_bounds(1, 2)
    ymin = corr3[0].min()
    ylim(ymin-0.1, 1.1)
    gca().spines['left'].set_bounds(ymin-0.1, 1.1)

    ylabel("Pearson r")
    xticks([1, 2], ['Chrimson', 'Ctrl'], fontsize=fontsize-1)

    # if i == 1: 
    #     yticks([])
    #     gca().spines["left"].set_visible(False)

    ############
    subplot(gs_corr3[1,0])
    simpleaxis(gca())
    xlim(0.5, 3)
    ylim(-0.5, 1)
    gca().spines['bottom'].set_bounds(1, 2)
    xlabel("minus baseline", labelpad=1)
    plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
    plot([2.2], [0], 'o', color=COLOR, markersize=0.5)


    tmp = [c-b for c, b in zip(corr3, base3)]
    vp = violinplot(tmp, showmeans=False, 
        showextrema=False, vert=True, side='high'
        )    
    for k, p in enumerate(vp['bodies']): p.set_color(cycle[k])

    m = [c.mean() for c in corr3]
    plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

    xticks([1,2],['',''])
    ylabel(r"Mean$\Delta$")
    # if i == 1: 
    #     yticks([])
    #     gca().spines["left"].set_visible(False)

    # COmputing tests    
    
    for i in range(2):
        zw, p = scipy.stats.mannwhitneyu(corr3[i], base3[i], alternative='greater')
        signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
        text(i+0.9, m[i]-0.07, s=map_significance[signi], va="center", ha="right")

    xl, xr = 2.5, 2.6
    plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
    plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
    plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
    zw, p = scipy.stats.mannwhitneyu(tmp[0], tmp[1])
    signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")








# # Raster
# subplot(gs2[0,0])
# simpleaxis(gca())
# for n in spikes.keys():
#     # cl = hsv_to_rgb([spikes.peaks[n]/(2*np.pi), 1, 1])
#     plot(spikes[n].restrict(ex).fillna(spikes.order[n]), '|', color=colors[st], markersize=2.1, mew=0.5)

# [axvspan(s, e, alpha=0.1) for s, e in opto_ep.intersect(ex).values]
# # ylim(-2, len(spikes)+2)
# xticks([])
# yticks([])

# # CMAP
# subplot(gs2[1,0])
# p = spikes.count(0.01, ex).smooth(0.04, size_factor=20)
# d=np.array([p.loc[i] for i in spikes.order.sort_values().index]).T
# p = nap.TsdFrame(t=p.t, d=d, time_support=p.time_support)
# p = np.sqrt(p / p.max(0))

# pcolormesh(p.index.values, 
#         np.arange(0, p.shape[1]),
#         p.values.T, cmap='jet')



# #####################################
# # CORRELATION
# #####################################
# data = cPickle.load(open(os.path.expanduser("~/Dropbox/LMNphysio/data/OPTO_SLEEP.pickle"), 'rb'))

# allr = data['allr']
# corr = data['corr']
# change_fr = data['change_fr']


# gs_corr = gridspec.GridSpecFromSubplotSpec(
#     2, 2, subplot_spec=gs_top[0, 2], hspace=0.4#, height_ratios=[0.5, 0.2, 0.2] 
# )


# subplot(gs_corr[0,:])

# plot(allr['adn']['opto']['ipsi']['wak'], allr['adn']['opto']['ipsi']['opto'], '.')
# plot(allr['adn']['opto']['ipsi']['wak'], allr['adn']['opto']['ipsi']['opto'], '.')
# xlim(-1, 1)
# ylim(-1, 1)
# gca().set_aspect("equal")

# ####################################
# # Pearson per sesion SLEEP
# ####################################
# subplot(gs_corr[1,1])
# simpleaxis(gca())

# orders = [('adn', 'opto', 'ipsi', 'opto'), 
#             ('adn', 'opto', 'ipsi', 'sws'), 
#             # ('adn', 'ctrl', 'ipsi', 'opto')
#             ]

# corr3 = []
# for j, keys in enumerate(orders):
#     st, gr, sd, k = keys

#     corr2 = corr[st][gr][sd]
#     corr2 = corr2[corr2['n']>4][k]
#     corr2 = corr2.values.astype(float)
#     corr3.append(corr2)

#     plot(corr2, np.abs(np.random.randn(len(corr2)))*0.1+np.ones(len(corr2))*(j+1+0.05), '.', markersize=1)


# violinplot(corr3, showmeans=True, side='low', showextrema=False, vert=False)
# # boxplot(corr3, widths=0.1, vert=True, positions=np.arange(1, len(corr3) + 1), showfliers=False)
# ylim(0.5, 2.5)
# # xmin = corr3[0].min()
# xmin = 0.0
# xlim(xmin, 1)
# # xlabel("Pearson r")
# yticks([1, 2], ['Chrimson', 'Control'])
# title("Sleep")
# xticks([])

# # ####################################
# # # Pearson per sesion WAKE
# # ####################################
# # data = cPickle.load(open(os.path.expanduser("~/Dropbox/LMNphysio/data/OPTO_WAKE.pickle"), 'rb'))

# # allr = data['allr']
# # corr = data['corr']
# # change_fr = data['change_fr']

# # subplot(gs_corr[1,0])
# # simpleaxis(gca())

# # orders = [('adn', 'opto', 'ipsi', 'opto'), 
# #             ('adn', 'opto', 'ipsi', 'pre'), 
# #             # ('adn', 'ctrl', 'ipsi', 'opto')
# #             ]

# # corr3 = []
# # for j, keys in enumerate(orders):
# #     st, gr, sd, k = keys

# #     corr2 = corr[st][gr][sd]
# #     corr2 = corr2[corr2['n']>4][k]
# #     corr2 = corr2.values.astype(float)
# #     corr3.append(corr2)

# #     plot(corr2, np.abs(np.random.randn(len(corr2)))*0.1+np.ones(len(corr2))*(j+1+0.05), '.', markersize=1)


# # violinplot(corr3, showmeans=True, side='low', showextrema=False, vert=False)
# # # boxplot(corr3, widths=0.1, vert=True, positions=np.arange(1, len(corr3) + 1), showfliers=False)
# # ylim(0.5, 2.5)
# # xlim(xmin, 1)
# # xlabel("Pearson r")
# # yticks([1, 2], ['Chrimson', 'Control'])

# # title("Wakefulness")


# #####################################
# # Histo
# #####################################
# gs_bottom = gridspec.GridSpecFromSubplotSpec(
#     1, 3, subplot_spec=outergs[1, 0], width_ratios=[0.1, 0.3, 0.2], wspace=0.6
# )


# gs_histo = gridspec.GridSpecFromSubplotSpec(
#     1, 1, subplot_spec=gs_bottom[0, 0]#, height_ratios=[0.5, 0.2, 0.2] 
# )


# subplot(gs_histo[0,0])
# noaxis(gca())
# img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN-PSB-opto.png")
# imshow(img, aspect="equal")
# xticks([])
# yticks([])


# #####################################
# # Examples ADN Bilateral
# #####################################
# st = 'adn'

# gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, gs_bottom[0,1])

# s = "B2800/B2809/B2809-240904B"
# ex = nap.IntervalSet(3897.29, 3905.7)
# path = os.path.join(data_directory, "OPTO", s)

# spikes, position, eps, tuning_curves = load_opto_data(path, st)
# opto_ep = eps['opto_ep']

# # exex = nap.IntervalSet(ex.start[0] - 10, ex.end[0] + 10)
# # p = spikes.count(0.01, exex).smooth(0.04, size_factor=20)
# # d=np.array([p.loc[i] for i in spikes.order.sort_values().index]).T
# # p = nap.TsdFrame(t=p.t, d=d, time_support=p.time_support)
# # p = np.sqrt(p / p.max(0))


# # ax = subplot(gs2[0,0])
# # noaxis(gca())
# # tmp = p.restrict(ex)
# # d = gaussian_filter(tmp.values, 1)
# # tmp2 = nap.TsdFrame(t=tmp.index.values, d=d)

# # pcolormesh(tmp2.index.values, 
# #         np.arange(0, tmp2.shape[1]),
# #         tmp2.values.T, cmap='GnBu', antialiased=True)
# # yticks([])


# subplot(gs2[1,0])
# simpleaxis(gca())
# for n in spikes.keys():
#     # cl = hsv_to_rgb([spikes.peaks[n]/(2*np.pi), 1, 1])
#     plot(spikes[n].restrict(ex).fillna(spikes.order[n]), '|', color=colors[st], markersize=2.1, mew=0.5)

# [axvspan(s, e, alpha=0.1) for s, e in opto_ep.intersect(ex).values]
# # ylim(-2, len(spikes)+2)
# xticks([])
# yticks([])

# #####################################
# # CORRELATION
# #####################################
# data = cPickle.load(open(os.path.expanduser("~/Dropbox/LMNphysio/data/OPTO_SLEEP.pickle"), 'rb'))

# allr = data['allr']
# corr = data['corr']
# change_fr = data['change_fr']


# gs_corr = gridspec.GridSpecFromSubplotSpec(
#     2, 1, subplot_spec=gs_bottom[0, 2], hspace=0.4#, height_ratios=[0.5, 0.2, 0.2] 
# )



# ####################################
# # Pearson per sesion SLEEP
# ####################################


# subplot(gs_corr[0,0])
# simpleaxis(gca())

# orders = [('adn', 'opto', 'bilateral', 'opto'), 
#             ('adn', 'opto', 'bilateral', 'sws'), 
#             # ('adn', 'ctrl', 'ipsi', 'opto')
#             ]

# corr3 = []
# for j, keys in enumerate(orders):
#     st, gr, sd, k = keys

#     corr2 = corr[st][gr][sd]
#     corr2 = corr2[corr2['n']>4][k]
#     corr2 = corr2.values.astype(float)
#     corr3.append(corr2)

#     plot(corr2, np.abs(np.random.randn(len(corr2)))*0.1+np.ones(len(corr2))*(j+1+0.05), '.', markersize=1)


# violinplot(corr3, showmeans=True, side='low', showextrema=False, vert=False)
# # boxplot(corr3, widths=0.1, vert=True, positions=np.arange(1, len(corr3) + 1), showfliers=False)
# ylim(0.5, 2.5)
# xmin = corr3[0].min()
# # xmin = 0.0
# xlim(xmin, 1)
# # xlabel("Pearson r")
# yticks([1, 2], ['Chrimson', 'Control'])
# title("Sleep")
# xticks([])

# ####################################
# # Pearson per sesion WAKE
# ####################################
# data = cPickle.load(open(os.path.expanduser("~/Dropbox/LMNphysio/data/OPTO_WAKE.pickle"), 'rb'))

# allr = data['allr']
# corr = data['corr']
# change_fr = data['change_fr']

# subplot(gs_corr[1,0])
# simpleaxis(gca())

# orders = [('adn', 'opto', 'bilateral', 'opto'), 
#             ('adn', 'opto', 'bilateral', 'pre'), 
#             # ('adn', 'ctrl', 'ipsi', 'opto')
#             ]

# corr3 = []
# for j, keys in enumerate(orders):
#     st, gr, sd, k = keys

#     corr2 = corr[st][gr][sd]
#     corr2 = corr2[corr2['n']>4][k]
#     corr2 = corr2.values.astype(float)
#     corr3.append(corr2)

#     plot(corr2, np.abs(np.random.randn(len(corr2)))*0.1+np.ones(len(corr2))*(j+1+0.05), '.', markersize=1)


# violinplot(corr3, showmeans=True, side='low', showextrema=False, vert=False)
# # boxplot(corr3, widths=0.1, vert=True, positions=np.arange(1, len(corr3) + 1), showfliers=False)
# ylim(0.5, 2.5)
# xlim(xmin, 1)
# xlabel("Pearson r")
# yticks([1, 2], ['Chrimson', 'Control'])

# title("Wakefulness")


outergs.update(top=0.96, bottom=0.09, right=0.98, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/fig3.pdf",
    dpi=200,
    facecolor="white",
)
# show()

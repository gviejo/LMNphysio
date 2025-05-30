# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   gviejo
# @Last Modified time: 2025-05-29 21:22:21
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
    fig_height = fig_width * golden_mean * 0.75  # height in inches
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

rcParams["font.family"] = 'Liberation Sans'
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




###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(1, 1, hspace = 0.5)


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

    
    gs2_ex = gridspec.GridSpecFromSubplotSpec(2, 1, gs2[i,0], hspace = 0.15, height_ratios=[1, 0.8])
    
    subplot(gs2_ex[0,0])
    simpleaxis(gca())    
    
    ms = [0.7, 0.9]
    plot(spikes.to_tsd("order").restrict(ex), '|', color=colors[st], markersize=ms[i], mew=0.25)


    s, e = opto_ep.intersect(ex).values[0]
    rect = patches.Rectangle((s, len(spikes)+1), width=e-s, height=1,
        linewidth=0, facecolor=opto_color)
    gca().add_patch(rect)
    [axvline(t, color=COLOR, linewidth=0.1, alpha=0.5) for t in [s,e]]

    ylim(0, len(spikes)+2)
    xlim(ex.start[0], ex.end[0])
    xticks([])
    yticks([0, len(spikes)-1], [1, len(spikes)])
    gca().spines['left'].set_bounds(0, len(spikes)-1)
    gca().spines['bottom'].set_bounds(s, e)
    title(name)
    ylabel("LMN")
    

    #
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
    
    gca().spines['bottom'].set_bounds(s, e)
    xticks([s, e], ['', ''])
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
    gs_corr2 = gridspec.GridSpecFromSubplotSpec(2,1, gs2[i,1], hspace=1)#, width_ratios=[0.2, 0.1])

    subplot(gs_corr2[0,0])
    simpleaxis(gca())

    s, e = ranges[f][1], ranges[f][2]
    rect = patches.Rectangle((s, 1.8), width=e-s, height=0.2,
        linewidth=0, facecolor=opto_color)
    gca().add_patch(rect)
    [axvline(t, color=COLOR, linewidth=0.1, alpha=0.5) for t in [s,e]]

    keys = orders[0]
    tmp = allfr[keys[0]][keys[1]][keys[2]]
    tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1.5, mode='constant'))
    tmp = tmp.loc[ranges[f][0]:ranges[f][-1]]
    m = tmp.mean(1)
    s = tmp.std(1)
    
    # plot(tmp, color = 'grey', alpha=0.2)
    plot(tmp.mean(1), color = COLOR, linewidth=0.75)
    fill_between(m.index.values, m.values-s.values, m.values+s.values, color=COLOR, alpha=0.2, ec=None)
    
    xlabel("Opto. time (s)", labelpad=1)
    xlim(ranges[f][0], ranges[f][-1])
    # ylim(0.0, 4.0)
    title(titles[i])
    xticks([ranges[f][1], ranges[f][2]])
    ylim(0, 2)    
    ylabel("Rate\n(norm.)")

    subplot(gs_corr2[1,0])
    simpleaxis(gca())

    # ch_fr = change_fr[keys[0]][keys[1]][keys[2]]['opto']

    chfr = change_fr[keys[0]][keys[1]][keys[2]]
    delta = (chfr['opto'] - chfr['pre'])/chfr['pre']    

    num_bins = 30
    bins = np.linspace(-1, 1, num_bins + 1)
    
    hist(delta.values, bins=bins, histtype="stepfilled", facecolor="#404040", edgecolor="#404040")
    axvline(0, color=COLOR, linestyle='--', linewidth=0.5)

    ylabel("%", labelpad=1)
    xlabel("% mod. rate", labelpad=1)
    ylim(0, 30)

    zw, p = scipy.stats.ttest_1samp(np.array(delta.values).astype(float), popmean=0)
    signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    text(0.6, 0.8, s=map_significance[signi], va="center", ha="left", transform=gca().transAxes)


    colors2 = [opto_color, "#FF7F50"]

    if f == 'OPTO_SLEEP':
        ################
        # CHRIMSON VS TDTOMATO
        ################
        gs_corr3 = gridspec.GridSpecFromSubplotSpec(2,2, gs2[i,2:], hspace=0.6, wspace=0.05)
        

        #
        ax1 = subplot(gs_corr3[0,0])
        simpleaxis(gca())
        title("Sessions")        
        gca().spines['bottom'].set_bounds(1, 2)

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

            plot(np.random.randn(len(corr2))*0.05+np.ones(len(corr2))*(j+1), corr2, '.', markersize=1, color=colors2[j])
        
            
        
        ymin = corr3[0].min()
        xlim(0.5, 3)
        ylim(ymin-0.1, 1.1)
        gca().spines['left'].set_bounds(ymin-0.1, 1.1)

        ylabel("Pearson r")
        xticks([1, 2], ['Chrimson', 'TdTomato'], fontsize=fontsize-1)

        #
        ax2 = subplot(gs_corr3[1,0])
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
        for k, p in enumerate(vp['bodies']): 
            p.set_color(colors2[k])
            p.set_alpha(1)

        m = [c.mean() for c in corr3]
        plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

        xticks([1,2],['',''])
        ylabel(r"Mean$\Delta$")

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

    
        ###############################
        # CHRIMSON CHRIMSON COMPARAISON
        ###############################
        orders2 =   [('adn', 'opto', 'bilateral', 'opto'),
                    ('adn', 'opto', 'bilateral', 'decimated')]

        #
        ax1 = subplot(gs_corr3[0,1])
        simpleaxis(gca())
        title("Matching FR")
        gca().spines['bottom'].set_bounds(1, 2)
        gca().spines['left'].set_visible(False)
        yticks([])

        corr3 = []
        base3 = []    
        for j, keys in enumerate(orders2):
            st, gr, sd, k = keys

            corr2 = corr[st][gr][sd]
            corr2 = corr2[corr2['n']>4][k]
            idx = corr2.index.values
            corr2 = corr2.values.astype(float)
            corr3.append(corr2)
            base3.append(baseline[st][gr][sd][k].loc[idx].values.astype(float))

            # plot(np.random.randn(len(corr2))*0.05+np.ones(len(corr2))*(j+1), corr2, '.', markersize=1)
        corr3 = pd.DataFrame(data=np.array(corr3).T, columns=['opto', 'decimated'])
        base3 = pd.DataFrame(data=np.array(base3).T, columns=['opto', 'decimated'])

        for s in corr3.index:
            plot([1, 2], corr3.loc[s,['opto', 'decimated']], '-', color=COLOR, linewidth=0.1)
        plot(np.ones(len(corr3)),   corr3['opto'], 'o', color=opto_color, markersize=1)
        plot(np.ones(len(corr3))*2, corr3['decimated'], 'o', color=COLOR, markersize=1)

        
        ymin = corr3['opto'].min()
        ylim(ymin-0.1, 1.1)
        xlim(0.5, 3)
        # gca().spines['left'].set_bounds(ymin-0.1, 1.1)

        # ylabel("Pearson r")
        xticks([1, 2], ['Chrimson', 'Control'], fontsize=fontsize-1)

        #
        ax2 = subplot(gs_corr3[1,1])
        simpleaxis(gca())
        gca().spines['left'].set_visible(False)
        yticks([])        
        ylim(-0.5, 1)
        xlim(0.5, 3)
        gca().spines['bottom'].set_bounds(1, 2)
        xlabel("minus baseline", labelpad=1)
        plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
        plot([2.2], [0], 'o', color=COLOR, markersize=0.5)


        tmp = corr3 - base3        
        vp = violinplot(tmp, showmeans=False, 
            showextrema=False, vert=True, side='high'
            )
        colors4 = [opto_color, COLOR]
        for k, p in enumerate(vp['bodies']): 
            p.set_color(colors4[k])
            p.set_alpha(1)

        m = tmp.mean(0).values
        plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

        xticks([1,2],['',''])
        # ylabel(r"Mean$\Delta$")

        # COmputing tests
        for i in range(2):
            zw, p = scipy.stats.mannwhitneyu(corr3.iloc[:,i], base3.iloc[:,i], alternative='greater')
            signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
            text(i+0.9, m[i]-0.07, s=map_significance[signi], va="center", ha="right")
        
        xl, xr = 2.5, 2.6
        plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
        plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
        plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
        zw, p = scipy.stats.mannwhitneyu(tmp['opto'].dropna(), tmp['decimated'].dropna())
        signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
        text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")



    if f == 'OPTO_WAKE':

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

            plot(np.random.randn(len(corr2))*0.05+np.ones(len(corr2))*(j+1), corr2, '.', markersize=1, color=colors2[j])
        
        
        xlim(0.5, 3)
        gca().spines['bottom'].set_bounds(1, 2)
        ymin = corr3[0].min()
        ylim(ymin-0.1, 1.1)
        gca().spines['left'].set_bounds(ymin-0.1, 1.1)

        ylabel("Pearson r")
        xticks([1, 2], ['Chrimson', 'TdTomato'], fontsize=fontsize-2)

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
        for k, p in enumerate(vp['bodies']): 
            p.set_color(colors2[k])
            p.set_alpha(1)

        m = [c.mean() for c in corr3]
        plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

        xticks([1,2],['',''])
        ylabel(r"Mean$\Delta$")

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

        ########################
        # TUNING CURVES WAKE OPTO
        #######################
        gs2_tc = gridspec.GridSpecFromSubplotSpec(2, 1, gs2[0,-1], hspace = 1.0, wspace=0.5)
        
        subplot(gs2_tc[0,0])
        simpleaxis(gca())

        lstyle = ['-', '--']

        st, gr, sd, k = orders[0]

        index = np.intersect1d(
            data['allsi'][st][gr][sd]['pre'][data['allsi'][st][gr][sd]['pre']>0.2].index.values,
            data['allsi'][st][gr][sd]['opto'][data['allsi'][st][gr][sd]['opto']>0.2].index.values
            )

        tcs = {}
        peaks = {}

        for j, k in enumerate(['opto', 'pre']):
            alltc = data['alltc'][st][gr][sd][k] # OPTO            
            peaks[k] = alltc.idxmax()
            alltc = centerTuningCurves_with_peak(alltc[index])
            tcs[k] = alltc

        tcs2 = {}
        tcs2['opto'] = tcs['opto']/tcs['pre'].max()
        tcs2['pre'] = tcs['pre']/tcs['pre'].max()
        
        colors3 = [opto_color, COLOR]
        for j, k in enumerate(['opto', 'pre']):
            plot(tcs2[k].mean(1), linewidth = 0.7, color = colors3[j], linestyle=lstyle[j])

        xlabel("Centered HD (deg.)")
        ylabel("Rate (norm.)")
        title("LMN-HD")
        xticks([-np.pi, 0, np.pi], [-180, 0, 180])
        yticks([0, 1], [0, 1])

        subplot(gs2_tc[1,0])
        simpleaxis(gca())

        plot([1,3.4],[0,0], linestyle='--', color=COLOR, linewidth=0.1, zorder=-100, alpha=0.2)
        plot([3.4], [0], 'o', color=COLOR, markersize=0.5)        

        # Width
        widths = {}
        for k in ['opto', 'pre']:
            tmp = tcs[k]/tcs[k].loc[0]
            tmp = tmp - 0.5
            widths[k] = np.rad2deg(np.abs(np.abs(tmp.loc[:0]).idxmin()) + np.abs(np.abs(tmp.loc[0:]).idxmin()))
        widths = pd.DataFrame(widths)
        # Peak diff
        maxpeaks = pd.DataFrame({"opto":tcs['opto'].max(), "pre":tcs['pre'].max()})
        # Peaks
        peaks = pd.DataFrame(peaks)        

        tmp = [(df['opto']-df['pre'])/(df['opto']+df['pre']) for df in [widths, maxpeaks]]

        d = peaks['opto'] - peaks['pre']
        d[d>np.pi] = 2*np.pi - d[d>np.pi]
        d[d<-np.pi] = d[d<-np.pi] + 2*np.pi
        d = d/np.pi

        tmp.append(d)


        vp = violinplot(tmp, showmeans=False, 
            showextrema=False, vert=True, side='high'
            )
        for k, p in enumerate(vp['bodies']): 
            p.set_color(opto_color)
            p.set_zorder(100)
            p.set_alpha(1)


        ylabel("Mean $\Delta$\nnorm.")
        xticks([1, 2, 3], ["Width", "Max FR", r"$\Delta$ ang."], rotation=45, ha='right')
        gca().spines['bottom'].set_bounds(1, 3)
        xlim(0.5, 3.5)

        # COmputing tests        
        for i, df in enumerate([widths, maxpeaks, peaks]):
            # if i == 2:
            #     p = ct.rayleigh(d*np.pi)
            # else:
            zw, p = scipy.stats.wilcoxon(df['pre'], df['opto'])
            signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
            m = tmp[i].mean()
            text(i+0.9, m-0.07, s=map_significance[signi], 
                va="center", ha="right", fontsize=fontsize-1,
                bbox=dict(facecolor='white', edgecolor='none',boxstyle='round,pad=0.01')
                )




outergs.update(top=0.92, bottom=0.09, right=0.98, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/fig3.pdf",
    dpi=200,
    facecolor="white",
)
# show()

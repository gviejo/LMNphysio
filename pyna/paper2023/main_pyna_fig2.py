# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-02-21 18:29:12
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
    fig_height = fig_width * golden_mean * 1.1  # height in inches
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
rcParams["xtick.major.size"] = 0
rcParams["ytick.major.size"] = 0
rcParams["xtick.major.width"] = 0
rcParams["ytick.major.width"] = 0
rcParams["axes.linewidth"] = 0.25
rcParams["axes.edgecolor"] = COLOR
rcParams["axes.axisbelow"] = True
rcParams["xtick.color"] = COLOR
rcParams["ytick.color"] = COLOR
rcParams['xtick.major.pad']=2
rcParams['ytick.major.pad']=2


colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}




clrs = [colors['psb'], 'darkgray']
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

mks = 3.1
alp = 1
medw = 1.3


fig = figure(figsize=figsize(2))

outergs = GridSpec(3, 1, figure=fig, hspace=0.6, height_ratios=[0.2, 0.2, 0.2])

#####################################
# PSB OPTO SLEEP
#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], 
    hspace=0.5, wspace = 0.3,
    width_ratios=[0.1, 0.4, 0.1]
)

# Histology
subplot(gs1[0, 0])
noaxis(gca())
img = mpimg.imread(dropbox_path+"/PSBopto.png")
imshow(img, aspect="equal", cmap="viridis")
xticks([])
yticks([])



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
        tmp = tmp.rolling(window=100, win_type='gaussian', center=True, min_periods=1, axis = 0).mean(std=1)
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


#####################################
# LMN OPTO
#####################################

lmnex = cPickle.load(open(dropbox_path + "/DATA_FIG_LMN_SLEEP_A8047-230310A.pickle", "rb"))

lmndata = {
    "sleep":cPickle.load(open(dropbox_path + "/OPTO_LMN_sleep.pickle", "rb")),
    "wake":cPickle.load(open(dropbox_path + "/OPTO_LMN_wake.pickle", "rb"))
    }


gs2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[1, 0], width_ratios=[0.05, 0.4, 0.1], hspace=0.5,
    wspace = 0.4
)

subplot(gs2[0, 0])
noaxis(gca())
img = mpimg.imread(dropbox_path+"/LMNopto.png")
imshow(img, aspect="equal", cmap="viridis")
xticks([])
yticks([])


#####################################
# LMN EXAMPLE
#####################################

#################################
# LMN tuning curves
#################################
gs2_1 = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=gs2[0, 1], hspace=0.5, wspace = 0.3,
    height_ratios=[0.1, 0.9, 0.1], width_ratios=[0.1, 0.8]
)

tuning_curves = lmnex['tc']

# All directions as arrows
subplot(gs2_1[1, 0], aspect="equal")
gca().spines["left"].set_position(("data", 0))
gca().spines["bottom"].set_position(("data", 0))
gca().spines["top"].set_visible(False)
gca().spines["right"].set_visible(False)

theta = tuning_curves.idxmax().values
radius = np.sqrt(tuning_curves.max(0).values)
for t, r in zip(theta, radius):
    arrow(
        0, 0, np.cos(t), np.sin(t), linewidth=1, color=colors['lmn'], width=0.01
    )

xticks([])
# xlim(-1, 1)
# ylim(-1, 1)
yticks([])
title("Pref. HD")


#################################
# LMN Raster
#################################
exopto_ep = lmnex['exopto_ep']

# gs_raster = gridspec.GridSpecFromSubplotSpec(3,2, 
#     subplot_spec = gs2_1[1,1], hspace=0.2,
#     height_ratios=[0.1, 0.5, 0.1]
#     )
subplot(gs2_1[1,1])
simpleaxis(gca())

spikes = lmnex['spikes']
exs = lmnex['ex']
exs = nap.IntervalSet(start=exs.get_intervals_center(0.3).index[0], end = exs.get_intervals_center(0.9).index[0])

# plot(spikes.to_tsd("order"), '|', color = colors['lmn'], markersize = mks, markeredgewidth = medw, alpha = 1)
order = spikes.get_info("order").values

for k, n in enumerate(order):
    spk = spikes[n].restrict(exs).index.values
    if len(spk):
        clr = clrs[1]
        plot(spk, np.ones_like(spk)*k, '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.8)


axvspan(exopto_ep.loc[0,'start'], exopto_ep.loc[0,'end'], 
    color = 'lightcoral', alpha=0.25,
    linewidth =0
    )
yticks([len(spikes)-1], [str(len(spikes))])
xlim(exs.loc[0,'start'], exs.loc[0,'end'])
xticks([])
xlabel(str(int(exs.tot_length('s')))+' s', horizontalalignment='right')#, x=1.0)
ylabel("LMN", rotation=0, labelpad=8, y=0.3)
title("Optogenetic inactivation of PSB")
xlabel("nREM sleep")

# # Tunning curves centerered
# subplot(gs2_1[1, 0])
# simpleaxis(gca())
# tc = centerTuningCurves2(tuning_curves)
# tc = tc / tc.loc[0]
# plot(tc, linewidth=0.1, color=colors["lmn"], alpha=0.4)
# plot(tc.mean(1), linewidth=0.5, color=colors["lmn"])
# xticks([])
# if i == 1:
#     xticks([-np.pi, 0, np.pi], [-180, 0, 180])
#     xlabel("Centered HD")

##################################
# LMN FIRING RATE
# for i, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(-4,14), slice(-1,2)], [slice(-4,0), slice(-1,0)]):
gs2_4 = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs2[0, 2], hspace=1
    )

ep = 'sleep'
sl = slice(-1, 2)
allmeta = lmndata[ep]['allmeta']
allfr = lmndata[ep]['allfr']
order = allmeta.sort_values(by="SI").index.values
tmp = allfr[order].loc[sl]
tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
subplot(gs2_4[0,0])
simpleaxis(gca())
plot(tmp, color = colors['lmn'], linewidth=0.1, alpha=0.25)
plot(tmp.mean(1), color = colors['lmn'], linewidth=1)
ylim(0, 2)
yticks([0, 2])
title('nREM')
axvspan(0, 1, color = 'lightcoral', alpha=0.2, ec = None)
xticks([0, 1])
ylabel("Firing\nrate (norm.)")
xlabel("Time (s)")

subplot(gs2_4[1,0])
simpleaxis(gca())

y = tmp.loc[0:1].mean().values-tmp.loc[-1:0].mean().values

hist(y, edgecolor=COLOR, facecolor='white')
xlabel("% mod")
ylabel("Count")
xlim(-1, 1)
xticks([-0.5, 0, 0.5], [-50, 0, 50])



#####################################
# LMN CORRELATION
#####################################
gs3 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[2, 0], 
    width_ratios=[0.5, 0.5], wspace=0.3
    )

allr = lmndata['sleep']['allr']
corr = lmndata['sleep']['corr']
epochs = ['sws', 'opto']
fcolors = [colors['lmn'], 'None']
labels = ['nREM (r)', 'Stim nREM (r)']

gs3_1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs3[0, 0], wspace = 1, width_ratios=[0.3, 0.3, 0.25]
    )

allaxis = []

for i, e in enumerate(epochs):
    subplot(gs3_1[0,i])
    simpleaxis(gca())
    gca().set_box_aspect(1)
    plot(allr['wak'], allr[e], 'o', color = fcolors[i], alpha = 1, markersize=2, mec=colors['lmn'])
    m, b = np.polyfit(allr['wak'].values, allr[e].values, 1)
    x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
    plot(x, x*m + b, color = COLOR, linewidth=1)
    xlabel('Wake (r)')
    ylabel(labels[i])
    xlim(allr['wak'].min(), allr['wak'].max())
    ylim(allr.iloc[:,1:].min().min(), allr.iloc[:,1:].max().max())
    r, p = scipy.stats.pearsonr(allr['wak'], allr[e])
    title('r = '+str(np.round(r, 3)))
    xticks([0, 0.5])
    yticks([0, 0.5])
    allaxis.append(gca())

subplot(gs3_1[0,2])
simpleaxis(gca())
corr = corr[corr['n']>4]
for i, e in enumerate(epochs):
    plot(np.random.randn(len(corr))*0.05+np.ones(len(corr))*i, corr[e], 
        'o', markersize=4,
        color = fcolors[i],
        mec=colors['lmn']
        )
    plot(
        [i - 0.2, i + 0.2],
        [corr[e].mean(), corr[e].mean()],
        "-",
        color=COLOR,
        linewidth=0.75,
    )
ylim(corr.min().min()-0.05, 1.05)
xticks([0, 1], ['nREM', 'PSB\ninh.'])
ylabel("Pair. corr (r)")
yticks([0, 1])
title("n>4")

xlims = []
ylims = []
for ax in allaxis:
    xlims.append(ax.get_xlim())
    ylims.append(ax.get_ylim())
xlims = np.array(xlims)
ylims = np.array(ylims)
xl = (np.min(xlims[:, 0]), np.max(xlims[:, 1]))
yl = (np.min(ylims[:, 0]), np.max(ylims[:, 1]))
for ax in allaxis:
    ax.set_xlim(xl)
    ax.set_ylim(yl)


#####################################
# LMN TUNING CURVES WAKE
#####################################
gs3_2 = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs3[0, 1], wspace=1, hspace=1
    )

# FIRING rate during wake
# LMN FIRING RATE
# for i, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(-4,14), slice(-1,2)], [slice(-4,0), slice(-1,0)]):
ep = 'wake'
sl = slice(-4, 14)
allmeta = lmndata[ep]['allmeta']
allfr = lmndata[ep]['allfr']
# order = allmeta.sort_values(by="SI").index.values
tmp = allfr.loc[sl]
tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
subplot(gs3_2[0,0])
simpleaxis(gca())
plot(tmp, color = colors['lmn'], linewidth=0.1, alpha=0.25)
plot(tmp.mean(1), color = colors['lmn'], linewidth=1)
ylim(0, 2)
yticks([0, 2])
title('Wake')
axvspan(0, 10, color = 'lightcoral', alpha=0.2, ec = None)
xticks([0, 10])
ylabel("Firing\nrate (norm.)")
xlabel("Time (s)", labelpad=1)

subplot(gs3_2[1,0])
simpleaxis(gca())

y = tmp.loc[0:10].mean().values-tmp.loc[-1:0].mean().values

hist(y, edgecolor=COLOR, facecolor='white', bins = np.linspace(-1, 1, 15))
xlabel("% mod")
ylabel("Count")
xlim(-0.75, 0.75)
xticks([-0.5, 0, 0.5], [-50, 0, 50])



# Tuning curves LMN
subplot(gs3_2[:,1])
tcn = lmndata[ep]['alltcn']
tco = lmndata[ep]['alltco']
peaks = {'wake':pd.Series(index=tcn.columns,data = np.array([circmean(tcn.index.values, tcn[i].values) for i in tcn.columns])),
        'opto':pd.Series(index=tco.columns,data = np.array([circmean(tco.index.values, tco[i].values) for i in tco.columns]))
        }
tcn = centerTuningCurves(tcn)
tco = centerTuningCurves(tco)
tcn = tcn / tcn.loc[0]
tco = tco / tco.loc[0]
simpleaxis(gca())
plot(tcn.mean(1), linewidth=1, color=clrs[i])
plot(tco.mean(1), '--', linewidth=1, color=clrs[i])
yticks([0, 1], ['0', '1'])
title("LMN-HD")
xticks([-np.pi, 0, np.pi], [-180, 0, 180])
xlabel("Centered HD", labelpad=1)
ylabel("Firing rate (norm.)")

subplot(gs3_2[0,2])
simpleaxis(gca())
peaks = pd.DataFrame.from_dict(peaks)
diff = np.abs(peaks['wake'] - peaks['opto'])
hist(diff,np.linspace(0, np.pi, 10), facecolor="white", edgecolor=COLOR)
xlim(0, np.pi)
xticks([0, np.pi], [0, 180])
title("Ang. diff")
ylabel("Count")

subplot(gs3_2[1,2])
simpleaxis(gca())
wn = np.rad2deg((np.abs(tcn[0:]-0.5)).idxmin())
wo = np.rad2deg((np.abs(tco[0:]-0.5)).idxmin())
widths = pd.DataFrame.from_dict({'wake':wn,'opto':wo})
for i, e in enumerate(widths.columns):
    plot(np.random.randn(len(widths))*0.08+np.ones(len(widths))*i, widths[e], 
        'o', markersize=4,
        color = fcolors[i],
        mec=colors['lmn']
        )
    plot(
        [i - 0.2, i + 0.2],
        [widths[e].mean(), widths[e].mean()],
        "-",
        color=COLOR,
        linewidth=0.75,
    )

ylabel("Width (deg)")
xticks([0, 1], ['Wake', 'Opto.'])
# yticks([0, ])
ylim(0, 180)

outergs.update(top=0.95, bottom=0.1, right=0.98, left=0.08)

# sys.exit()

savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2023/fig2.pdf",
    dpi=200,
    facecolor="white",
)
# show()


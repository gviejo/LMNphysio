# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-08-18 12:39:15
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

data = cPickle.load(open(dropbox_path + "/DATA_FIG_LMN_ADN_A5043-230302A.pickle", "rb"))
decoding = {
    "wak": nap.Tsd(t=data["wak"].index.values, d=data["wak"].values, time_units="s"),
    "sws": nap.Tsd(t=data["sws"].index.values, d=data["sws"].values, time_units="s"),
    "rem": nap.Tsd(t=data["rem"].index.values, d=data["rem"].values, time_units="s"),
}
angle = data["angle"]
tcurves = data["tcurves"]
peaks = data["peaks"]
spikes = data["spikes"]
tokeep = data["tokeep"]
adn = data["adn"]
lmn = data["lmn"]

exs = {"wak": data["ex_wak"], "rem": data["ex_rem"], "sws": data["ex_sws"]}

# exs["wak"] = nap.IntervalSet(start=7968.0, end=7990.14)
# exs["sws"] = nap.IntervalSet(start=12695.73, end=12701.38)


###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(2))

outergs = GridSpec(3, 1, figure=fig, height_ratios=[0.2, 0.2, 0.3], hspace=0.45)

#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], width_ratios=[0.3, 0.3, 0.6], wspace=0.4
)

names = ["ADN", "LMN"]

#####################################
# Histo
#####################################
gs_histo = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs1[0, 0], height_ratios=[0.6, 0.4], wspace=0.01, hspace=0.4
)

subplot(gs_histo[0, :])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/CosyneData/brain_render1.png")
imshow(img, aspect="equal")
xticks([])
yticks([])

subplot(gs_histo[1, 0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/CosyneData/histo_adn.png")
imshow(img[:, :, 0], aspect="equal", cmap="viridis")
title("ADN")
xticks([])
yticks([])

subplot(gs_histo[1, 1])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/CosyneData/histo_lmn.png")
imshow(img[:, :, 0], aspect="equal", cmap="viridis")
title("LMN")
xticks([])
yticks([])


#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=gs1[0, 1], height_ratios=[0.5, 0.5, 0.25]
)

for i, (st, name) in enumerate(zip([adn, lmn], ["adn", "lmn"])):

    
    # Tunning curves centerered
    subplot(gs_tc[i, 0])
    simpleaxis(gca())
    tc = centerTuningCurves(tcurves[st])    
    tc = tc/tc.loc[0]
    plot(tc, linewidth=0.1, color = colors[name], alpha=0.4)
    plot(tc.mean(1), linewidth=0.5, color = colors[name])
    xticks([])
    if i == 1:
        xticks([-np.pi, 0, np.pi], [-180, 0, 180])
        xlabel("Centered HD")

    # All directions as arrows
    subplot(gs_tc[i, 1], aspect="equal")
    gca().spines["left"].set_position(("data", 0))
    gca().spines["bottom"].set_position(("data", 0))
    gca().spines["top"].set_visible(False)
    gca().spines["right"].set_visible(False)
    
    theta = peaks[st].values
    radius = np.sqrt(tcurves[st].max(0).values)
    for t, r in zip(theta, radius):
        arrow(0, 0, np.cos(t), np.sin(t), linewidth=0.05, color = colors[name], width=0.01)

    xticks([-1, 1], ["-pi", "pi"])
    # xlim(-1, 1)
    # ylim(-1, 1)

    xticks([])
    yticks([])



#########################
# RASTER PLOTS
#########################
gs_raster = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=gs1[0, 2], hspace=0.2, height_ratios=[0.5, 0.5, 0.4], wspace=0.1
)

mks = 1.1
alp = 1
medw = 0.08

epochs = ["Wake", "REM sleep", "nREM sleep"]


for i, ep in enumerate(["wak", "rem"]):
    for j, (st, name) in enumerate(zip([adn, lmn], ["adn", "lmn"])):
        subplot(gs_raster[j, i])
        simpleaxis(gca())        
        gca().spines["bottom"].set_visible(False)
        if i > 0:
            gca().spines["left"].set_visible(False)

        order = tcurves[st].idxmax().sort_values().index.values

        for k, n in enumerate(order):
            spk = spikes[n].restrict(exs[ep]).index.values
            if len(spk):
                clr = colors[name]
                plot(
                    spk,
                    np.ones_like(spk) * k,
                    "|",
                    color=clr,
                    markersize=mks,
                    markeredgewidth=medw,
                    alpha=alp,
                )
        if j == 0:
            title(epochs[i], pad=5)
        # ylim(0, 2*np.pi)
        xlim(exs[ep].loc[0, "start"], exs[ep].loc[0, "end"])
        xticks([])
        yticks([])
        if i == 0:
            yticks([len(st)], [len(st)+1])
        gca().spines["bottom"].set_visible(False)

        if i == 0:
            ylabel(name.upper(), rotation=0, labelpad=10)

    #########################
    # ANGLE
    #########################
    subplot(gs_raster[-1, i])
    simpleaxis(gca())
    
    gca().spines['bottom'].set_bounds(exs[ep].end[0] - 5, exs[ep].end[0])

    xticks([exs[ep].end[0] - 2.5], ["5 s"])

    if i > 0:
        gca().spines["left"].set_visible(False)    
    
    if i==0: 
        yticks([0, 2*np.pi], [0, 360])
    else:
        yticks([])
    p = data["p_" + ep].restrict(exs[ep]).values.T
    p[np.isnan(p)] = 0.0
    startend = exs[ep].values[0]    
    imshow(
        gaussian_filter(p, 3),
        origin="lower",
        aspect="auto",
        cmap="Greys",
        extent = (startend[0], startend[1], 0, 2*np.pi)
    )
    if i == 0:
        plot(angle.restrict(exs[ep]), linewidth=0.25, color = "seagreen", alpha=1)
        ylabel("HD", rotation=0, labelpad=10)


###############################
# GLM - HMM SWS
###############################
gs22 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[1, 0], width_ratios=[0.2, 0.2, 0.2], wspace=0.15
)  # , hspace = 0.5)

datahmm= cPickle.load(open(os.path.join(dropbox_path, 'GLM_HMM_A5043-230302A_18-08-2023.pickle'), "rb"))


#############################
# MAnifolds
gs_manifold = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs22[0, 0])


xy, _, _ = np.histogram2d(datahmm["imap"][:,0], datahmm["imap"][:,1], 20)
subplot(gs_manifold[0,0])
noaxis(gca())
imshow(xy, cmap="Greys")



xy, _, _ = np.histogram2d(datahmm["imapr"][:,0], datahmm["imapr"][:,1], 20)
subplot(gs_manifold[1,0])
noaxis(gca())
imshow(xy, cmap="Greys")


#############################
# Weights



gs_raster = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs22[0, 1], hspace=0.2, height_ratios=[0.5, 0.5, 0.4], wspace=0.1
)

ep = "sws"
for j, (st, name) in enumerate(zip([adn, lmn], ["adn", "lmn"])):
    subplot(gs_raster[j, 0])
    simpleaxis(gca())        
    gca().spines["bottom"].set_visible(False)
    gca().spines["left"].set_visible(False)

    order = tcurves[st].idxmax().sort_values().index.values

    for k, n in enumerate(order):
        spk = spikes[n].restrict(exs[ep]).index.values
        if len(spk):
            clr = colors[name]
            plot(
                spk,
                np.ones_like(spk) * k,
                "|",
                color=clr,
                markersize=mks,
                markeredgewidth=medw*2,
                alpha=alp,
            )
    if j == 0:
        title(epochs[i], pad=5)
    # ylim(0, 2*np.pi)
    xlim(exs[ep].loc[0, "start"], exs[ep].loc[0, "end"])
    xticks([])    
    yticks([len(st)], [len(st)+1])
    gca().spines["bottom"].set_visible(False)
    ylabel(name.upper(), rotation=0, labelpad=10)

subplot(gs_raster[-1, 0])
simpleaxis(gca())
gca().spines['bottom'].set_bounds(exs[ep].end[0] - 5, exs[ep].end[0])
xticks([exs[ep].end[0] - 2.5], ["5 s"])
yticks([0, 2*np.pi], [0, 360])
p = data["p_" + ep].restrict(exs[ep]).values.T
p[np.isnan(p)] = 0.0
startend = exs[ep].values[0]    
imshow(
    gaussian_filter(p, 3),
    origin="lower",
    aspect="auto",
    cmap="Greys",
    extent = (startend[0], startend[1], 0, 2*np.pi)
)


###############################
# Correlation
###############################
gs2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[2, 0], width_ratios=[0.3, 0.25], wspace=0.15
)  # , hspace = 0.5)

gscor = gridspec.GridSpecFromSubplotSpec(
    2,
    4,
    subplot_spec=gs2[0, 0],
    wspace=1.2,
    hspace=0.6,
    width_ratios=[0.08, 0.5, 0.5, 0.5],
)

# gscor2 = gridspec.GridSpecFromSubplotSpec(4,2, subplot_spec = gs2[0,1], height_ratios=[0.1, 0.1, 0.6, 0.2], hspace=0.01)

allaxis = []


paths = [
    dropbox_path + "/All_correlation_ADN.pickle",
    # dropbox_path+'/All_correlation_ADN_LMN.pickle',
    dropbox_path + "/All_correlation_LMN.pickle",
]
# names = ['ADN', 'ADN/LMN', 'LMN']
# clrs = ['lightgray', 'darkgray', 'gray']
# clrs = ['sandybrown', 'olive']
# clrs = ['lightgray', 'gray']
names = ["ADN", "LMN"]
corrcolor = COLOR
mkrs = 6

xpos = [0, 2]

for i, (p, n) in enumerate(zip(paths, names)):
    #
    data3 = cPickle.load(open(p, "rb"))
    allr = data3["allr"]
    # pearsonr = data3['pearsonr']
    print(n, allr.shape)
    print(
        len(
            np.unique(
                np.array(
                    [
                        [p[0].split("-")[0], p[1].split("-")[0]]
                        for p in np.array(allr.index.values)
                    ]
                ).flatten()
            )
        )
    )

    #############################################################
    subplot(gscor[i, 1])
    simpleaxis(gca())
    scatter(
        allr["wak"],
        allr["rem"],
        color=colors[n.lower()],
        alpha=0.5,
        edgecolor=None,
        linewidths=0,
        s=mkrs,
    )
    m, b = np.polyfit(allr["wak"].values, allr["rem"].values, 1)
    x = np.linspace(allr["wak"].min(), allr["wak"].max(), 5)
    r, p = scipy.stats.pearsonr(allr["wak"], allr["rem"])
    plot(
        x,
        x * m + b,
        color=corrcolor,
        label="r = " + str(np.round(r, 2)),
        linewidth=0.75,
    )
    if i == 1:
        xlabel("Wake corr. (r)")
    ylabel("REM corr. (r)")
    text(
        -1,
        0.5,
        n,
        horizontalalignment="center",
        verticalalignment="center",
        transform=gca().transAxes,
        fontsize=fontsize,
    )
    legend(
        handlelength=0.0,
        loc="center",
        bbox_to_anchor=(0.15, 0.9, 0.5, 0.5),
        framealpha=0,
    )
    ax = gca()
    # ax.set_aspect(1)
    locator_params(axis="y", nbins=3)
    locator_params(axis="x", nbins=3)
    # xlabel('Wake corr. (r)')
    allaxis.append(gca())

    #############################################################
    subplot(gscor[i, 2])
    simpleaxis(gca())
    scatter(
        allr["wak"],
        allr["sws"],
        color=colors[n.lower()],
        alpha=0.5,
        edgecolor=None,
        linewidths=0,
        s=mkrs,
    )
    m, b = np.polyfit(allr["wak"].values, allr["sws"].values, 1)
    x = np.linspace(allr["wak"].min(), allr["wak"].max(), 5)
    r, p = scipy.stats.pearsonr(allr["wak"], allr["sws"])
    plot(
        x,
        x * m + b,
        color=corrcolor,
        label="r = " + str(np.round(r, 2)),
        linewidth=0.75,
    )
    if i == 1:
        xlabel("Wake corr. (r)")
    ylabel("nREM corr. (r)")
    legend(
        handlelength=0.0,
        loc="center",
        bbox_to_anchor=(0.15, 0.9, 0.5, 0.5),
        framealpha=0,
    )
    # title(n, pad = 12)
    ax = gca()
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (
        ax.get_ylim()[1] - ax.get_ylim()[0]
    )
    # ax.set_aspect(ratio_default*aspectratio)
    # ax.set_aspect(1)
    locator_params(axis="y", nbins=3)
    locator_params(axis="x", nbins=3)
    allaxis.append(gca())

    #############################################################
    subplot(gscor[i, 3])
    simpleaxis(gca())
    marfcol = [colors[n.lower()], "white"]

    y = data3["pearsonr"]
    y = y[y["count"] > 6]
    for j, e in enumerate(["rem", "sws"]):
        plot(
            np.ones(len(y)) * j + np.random.randn(len(y)) * 0.1,
            y[e],
            "o",
            markersize=2,
            markeredgecolor=colors[n.lower()],
            markerfacecolor=marfcol[j],
        )
        plot(
            [j - 0.2, j + 0.2],
            [y[e].mean(), y[e].mean()],
            "-",
            color=corrcolor,
            linewidth=0.75,
        )
    xticks([0, 1])
    xlim(-0.4, 1.4)
    # print(scipy.stats.ttest_ind(y["rem"], y["sws"]))
    print(scipy.stats.wilcoxon(y["rem"], y["sws"]))

    ylim(0, 1.3)
    yticks([0, 1], [0, 1])
    # title(names[i], pad = 12)

    gca().spines.left.set_bounds((0, 1))
    ylabel("r")

    if i == 0:
        xticks([0, 1], ["", ""])
    else:
        xticks([0, 1], ["REM", "nREM"])
        text(
            0.5,
            -0.45,
            "vs Wake",
            horizontalalignment="center",
            verticalalignment="center",
            transform=gca().transAxes,
        )

    lwdtt = 0.1
    plot([0, 1], [1.2, 1.2], "-", color=COLOR, linewidth=lwdtt)
    plot([0, 0], [1.15, 1.2], "-", color=COLOR, linewidth=lwdtt)
    plot([1, 1], [1.15, 1.2], "-", color=COLOR, linewidth=lwdtt)
    TXT = ["n.s.", "p<0.001"]
    text(
        0.5,
        1.4,
        TXT[i],
        horizontalalignment="center",
        verticalalignment="center",
    )


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


###############################
# GLM cross corrs
###############################
gsglm = gridspec.GridSpecFromSubplotSpec(
    2,
    4,
    subplot_spec=gs2[0, 1],
    wspace=0.5,
    hspace=0.6,
    width_ratios=[0.1, 0.5, 0.5, 0.5],
)


# simpleGLM
dataglm = cPickle.load(
    open(
        os.path.join(
            os.path.expanduser("~") + "/Dropbox/CosyneData", "GLM_BETA_WITHIN.pickle"
        ),
        "rb",
    )
)

coefs_mua = dataglm["coefs_mua"]
coefs_pai = dataglm["coefs_pai"]
pairs_info = dataglm["pairs_info"]

clrs = ["sandybrown", "olive"]

for i, g in enumerate(["adn", "lmn"]):
    inters = np.linspace(0, np.pi, 5)
    idx = np.digitize(pairs_info[g]["offset"], inters) - 1

    # angular difference
    subplot(gsglm[i, 0])
    simpleaxis(gca())
    plot(
        pairs_info[g]["offset"].values.astype("float"),
        np.arange(len(pairs_info[g]))[::-1],
        color="grey",
    )
    for j, l in enumerate([0, 3]):
        plot(
            pairs_info[g]["offset"].values[idx == l],
            np.arange(len(pairs_info[g]))[::-1][idx == l],
            color=clrs[j],
        )
    axhspan(len(idx) - np.sum(idx == 0), len(idx), color=clrs[0], alpha=0.2)
    axhspan(0, np.sum(idx == 3), color=clrs[1], alpha=0.2)
    xticks([0, np.pi], ["0", r"$\pi$"])
    ylabel("Pair", labelpad=-10)
    # xticks([])
    yticks([len(idx)])

    # glm
    subplot(gsglm[i, 1])
    simpleaxis(gca())
    for j, l in enumerate([0, 3]):
        tmp = coefs_pai[g]["sws"].iloc[
            :, idx == l
        ]  # .rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        m = tmp.mean(1)
        s = tmp.sem(1)
        plot(m, "-", color=clrs[j], linewidth=1)
        fill_between(
            m.index.values,
            m.values - s.values,
            m.values + s.values,
            color=clrs[i],
            alpha=0.25,
        )
    yticks([])
    ylabel(r"$\beta_t^{nREM}$", labelpad=-1)
    if i == 1:
        xlabel("GLM time lag (s)")

    if i == 0:
        axip = gca().inset_axes([-0.1, 1.2, 1, 0.5])
        noaxis(axip)
        axip.annotate(
            "Pop.",
            xy=(0.7, 0.3),
            xytext=(0.1, 0.5),
            color=COLOR,
            arrowprops=dict(
                facecolor="green", headwidth=1.5, headlength=1, width=0.01, ec="grey"
            ),
            fontsize=6,
        )
        axip.annotate(
            "",
            xy=(0.7, 0.1),
            xytext=(0.4, 0.1),
            arrowprops=dict(
                facecolor=COLOR, headwidth=1.5, headlength=1, width=0.01, ec="grey"
            ),
        )
        axip.text(0.0, 0.0, "Unit", fontsize=6)
        axip.text(0.8, 0.0, "Unit", fontsize=6)
        axip.text(0.5, -0.25, r"$\beta_t$", fontsize=6)
        # axip.text(0.6, 0.5, r"$\beta_t^{P}$", fontsize = 6)

##########################
# LMN -> ADN GLM
##########################


outergs.update(top=0.96, bottom=0.09, right=0.96, left=0.025)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2023/fig1.pdf",
    dpi=200,
    facecolor="white",
)
# show()

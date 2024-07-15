# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-07-15 19:13:40
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

data = cPickle.load(open(dropbox_path + "/DATA_SUPP_FIG_1_HMM_exemple_nemos.pickle", "rb"))

W = data["W"]
scores = data["scores"]
A = data["A"]
bestA = data["bestA"]
Z = data["Z"]
bestZ = data["bestZ"]
tc = data["tc"]
tcr = data["tcr"]
Yt = data['Yt']
Y = data['Y']
Yr = data['Yr']
B = data['B']
B = B[:,::-1]
random_scores = data["random_scores"]
O = data['O']

###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(2))

outergs = GridSpec(2, 1, figure=fig, height_ratios=[0.2, 0.3], hspace=0.3)


#####################################
# GLM 
#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[0, 0], width_ratios=[0.4, 0.2], wspace=0.2
)


gs1_1 = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs1[0, 0], width_ratios=[0.2, 0.4], wspace=0.4, hspace=0.4
)


cmaps = ['gist_yarg', 'viridis']

for i, (t, y) in enumerate(zip([tc, tcr], [Y, Yr])):

    # Tuning curves

    subplot(gs1_1[i,0])
    im = imshow(t.T, cmap = cmaps[0], aspect='auto')

    if i == 0:
        title("Tuning curves")

        # colorbar(im, ax=gca())

    ylabel("GLM "+str(i+1), rotation=0, labelpad=15)

    xticks([0, 60])
    if i == 1:
        xlabel("Feature (deg)")

    # Activity

    subplot(gs1_1[i,1])
    imshow(y.T, cmap=cmaps[1], aspect='auto')

    if i == 0:
        title("Spikes count")
        ylabel("Units", y=0)        

    if i == 1:
        xlabel("Time")


#####################################
# Basis
#####################################
gs_1_2 = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs1[0, 1], height_ratios=[0.4, 0.4], 
    wspace=0.4, hspace=1.1
)

colors = ['blue', 'orange', 'green']


subplot(gs_1_2[0,:])
simpleaxis(gca())
for i in range(3):
    plot(B[:,i][::-1], color = colors[i], linewidth=0.5)
title("Basis function")
xticks([0, 50, 100], [-50,0,50])
xlabel("Time", labelpad=0)
#####################################
# Weights
#####################################
gs_1_3 = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_1_2[1,:], wspace=0.4
)

axes = [[], []]

for i in range(2):    

    w = W[i+1].reshape(12,3,12)[0:-1] # Wrong
    w = w[:,::-1,:]
    tmp = []
    for j in range(3):
        tmp.append(np.array([np.roll(w[:,j,k], n) for k, n in zip(np.arange(12), np.arange(-6, 6)[::-1])]).T)
    tmp = np.array(tmp)

    m = tmp.mean(-1).T
    s = tmp.std(-1).T

    if i==0:
        ymin = m.min(0)
        ymax = m.max(0)
    
    for j in range(m.shape[1]):
        subplot(gs_1_3[i,j])    
        simpleaxis(gca())
        t = np.insert(m[:,j], 5, 0)
        bar(np.arange(-6, 6), t, color = colors[j])

        yticks([])

        if j == 0:
            ylabel("GLM "+str(i+1), rotation=0, labelpad=15)

        if i==0 and j == 1:
            title("Weights")

        if i==1 and j == 1:
            xlabel(r"w_n->w_0")

        if i == 0:
            xticks([-1, 5], ['',''])
        if i == 1:
            xticks([-1, 5], [0, 6])
    




#####################################
# HMM 
#####################################
gs2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[1, 0], width_ratios=[0.8, 0.3], wspace=0.3
)


gs2_1 = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs2[0, 0], height_ratios=[0.4, 0.1, 0.1], wspace=0.4
)

start=250
end = 1000

subplot(gs2_1[0,0])
imshow(Yt[start:end].T, cmap = cmaps[1], aspect='auto')
xticks([])
ylabel("Units")
title("Spike counts")

subplot(gs2_1[1,0])
simpleaxis(gca())
plot(Z[start:end], color = 'grey')
xlim(0, end - start)
xticks([])
yticks([0, 1, 2])
ylabel("True\nstate", rotation=0, labelpad=20)

subplot(gs2_1[2,0])
simpleaxis(gca())
plot(bestZ[start:end], color = 'coral')
ylabel("Inferred\nstate", rotation=0, labelpad=20)
xlim(0, end - start)
yticks([0, 1, 2])
xlabel("Time")
# subplot(gs2_1[3,0])
# simpleaxis(gca())

# imshow(O.T, aspect='auto')

# ylabel("P(O)", rotation=0, labelpad=20)
# xlim(0, len(Yt))



#####################################
# Log likelihood
#####################################
gs2_2 = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=gs2[0, 1], hspace=1.0
)


subplot(gs2_2[0,:])
simpleaxis(gca())
for s in scores:
    plot(s[0:10], '-', linewidth=0.5)
ylabel("Log\nlikelihood", rotation=0, labelpad=20)
yticks([])


#####################################
# Transition matrix
#####################################


subplot(gs2_2[1,0])
imshow(A)
# title("Transition\nmatrix")
xticks([0, 1, 2])
yticks([0, 1, 2])
title("True")
ylabel("Transition", rotation=0, labelpad=20)

subplot(gs2_2[1,1])
imshow(bestA)
title("Inferred")
xticks([0, 1, 2])
yticks([0, 1, 2], ['', '', ''])


#####################################
# Scores
#####################################
subplot(gs2_2[2,:])
simpleaxis(gca())

hist(random_scores, label="Random")
axvline(np.sum(Z == bestZ)/len(Z), color = 'coral')

legend(frameon=False)
xlabel(r"% correct")




outergs.update(top=0.96, bottom=0.09, right=0.96, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2023/supp_fig1.pdf",
    dpi=200,
    facecolor="white",
)
# show()

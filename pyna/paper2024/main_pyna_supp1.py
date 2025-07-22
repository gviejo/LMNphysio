# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-07-19 15:01:09
# @Last Modified by:   gviejo
# @Last Modified time: 2025-07-21 22:51:02
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
    fig_height = fig_width * golden_mean * 1.5  # height in inches
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

cmap = plt.get_cmap("Set2")

###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"
data = cPickle.load(open(os.path.join(dropbox_path, 'All_TC.pickle'), 'rb'))

alltc = data['alltc']
allinfo = data['allinfo']



fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, hspace = 0.3, height_ratios=[0.2, 0.6])


#########################
# TUNING CURVES SELECTION
#########################

gs1 = gridspec.GridSpecFromSubplotSpec(1,3, outergs[0,0], 
    wspace = 0.5, width_ratios=[0.0, 0.5, 0.5]
    )

for i, st in enumerate(['adn', 'lmn']):
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
    ylabel("Mutual Information\n(bits/spk)", labelpad=5)

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

    ax = subplot(gs1_1[1,1])
    imshow(tc.values.T, aspect='auto')


################################
# REM Correlation
################################
gs2 = gridspec.GridSpecFromSubplotSpec(2,3, outergs[1,0], 
    wspace = 0.1, hspace=0.1, width_ratios=[0.25, 0.5, 0.5]
    )


gs_bottom_left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs2[0,0], hspace=0.8, height_ratios=[0.4, 0.4]
)


names = {'adn':"ADN", 'lmn':"LMN"}

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
r_lmn = data['pearsonr']
data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'rb'))
r_adn = data['pearsonr']
allr_sess = {'adn':r_adn, 'lmn':r_lmn}


subplot(gs_bottom_left[0,0])
simpleaxis(gca())

for i,g in enumerate(['adn', 'lmn']):

    tmp = allr_sess[g]['rem']
    plot(np.ones(len(tmp))*(i+1) + np.random.randn(len(tmp))*0.05, tmp.values,  'o', color = colors[g], markersize = 1)
    plot([i+1-0.2, i+1+0.2], [tmp.mean(), tmp.mean()], linewidth=1, color = 'grey')

xlim(0.5, 3)
gca().spines['bottom'].set_bounds(1, 2)
ylim(0, 1.1)
gca().spines['left'].set_bounds(0, 1.1)

ylabel("Pop. coherence (r)", y=0, labelpad=3)
xticks([1, 2], [names['adn'], names['lmn']])
title("Wake vs REM")


subplot(gs_bottom_left[1,0])
simpleaxis(gca())
xlim(0.5, 3)
ylim(-0.1, 1)
gca().spines['bottom'].set_bounds(1, 2)
xlabel("minus baseline", labelpad=1)
# if i == 1: gca().spines["left"].set_visible(False)
plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
plot([2.2], [0], 'o', color=COLOR, markersize=0.5)
tmp = [allr_sess[g]['rem'] for g in ['adn', 'lmn']]

vp = violinplot(tmp, showmeans=False, 
    showextrema=False, vert=True, side='high'
    )
for k, p in enumerate(vp['bodies']): 
    p.set_color(colors[['adn', 'lmn'][k]])
    p.set_alpha(1)

m = [a.mean() for a in tmp]
plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

xticks([1,2],['ADN','LMN'])
# ylabel(r"Mean$\Delta$")


# COmputing tests
map_significance = {
    1:"n.s.",
    2:"*",
    3:"**",
    4:"***"
}

# for i, g in enumerate(['adn', 'lmn']):
#     zw, p = scipy.stats.wilcoxon(pearson[k].values.astype("float"), baseline[k].values.astype("float"), alternative='greater')
#     signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
#     text(i+0.9, m[i]-0.07, s=map_significance[signi], va="center", ha="right")

xl, xr = 2.5, 2.6
plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
zw, p = mannwhitneyu(tmp[1], tmp[0])
signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")




################################
# PAIRWISE CORRELATION / SESSION
################################


for i, st in enumerate(['adn', 'lmn']):

    data = cPickle.load(open(os.path.join(dropbox_path, f'All_correlation_{st.upper()}.pickle'), 'rb'))
    allr = data['allr'] 
    sessions = [s.split("/")[-1] for s in data['pearsonr'].index.values]

    for k, e in enumerate(['sws', 'rem']):

        n = int(sqrt(len(sessions)))+1
        
        gs2_1 = gridspec.GridSpecFromSubplotSpec(n,n, gs2[i,k+1])

        ngrid = np.indices((n,n))
        ngrid = (ngrid[0].flatten(), ngrid[1].flatten())

        for j, s in enumerate(sessions):

            index = [k for k in allr.index.values if s in k[0]]

            ax = subplot(gs2_1[ngrid[0][j],ngrid[1][j]])
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)        
            gca().set_aspect("equal")
            scatter(allr.loc[index, 'wak'], allr.loc[index, e],
                    0.4,
                    color=colors[st]                    
                )
            xlim(-1, 1)
            ylim(-1,1)
            if ngrid[0][j]==2 and ngrid[1][j] == 0 and k==0:
                ylabel(st.upper(), labelpad=15)
                
            if ngrid[0][j]==0 and ngrid[1][j] == 3 and i == 0:
                title(e.upper())
                

            # else:
            xticks([])
            yticks([])

            m, b = np.polyfit(allr.loc[index, 'wak'].values, allr.loc[index, e].values, 1)
            x = np.linspace(allr.loc[index, 'wak'].values.min(), allr.loc[index, 'wak'].values.max(),5)
            plot(x, x*m + b, color=COLOR, linewidth=0.5)

            
            



outergs.update(top=0.96, bottom=0.02, right=0.95, left=0.08)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/supp1.pdf",
    dpi=200,
    facecolor="white",
)
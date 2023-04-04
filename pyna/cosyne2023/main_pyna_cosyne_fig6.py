# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 17:49:50
# @Last Modified by:   gviejo
# @Last Modified time: 2023-03-21 15:51:44
import numpy as np
import pandas as pd
import pynapple as nap

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import matplotlib.font_manager as font_manager
#matplotlib.style.use('seaborn-paper')
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.patches import Ellipse, FancyArrowPatch, ArrowStyle

import _pickle as cPickle
import hsluv

import os
import sys
from scipy.ndimage import gaussian_filter

from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap

# from umap import UMAP




def figsize(scale):
    fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
    #fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_width = 4.0
    fig_height = fig_width*golden_mean*1.3         # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)

font_dir = ['/home/guillaume/Dropbox/CosyneData/figures_poster_2022']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

fontsize = 7

COLOR = (0.25, 0.25, 0.25)

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = fontsize
rcParams['text.color'] = COLOR
rcParams['axes.labelcolor'] = COLOR
rcParams['axes.labelsize'] = fontsize
rcParams['axes.labelpad'] = 3
#rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = fontsize
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize
rcParams['figure.titlesize'] = fontsize
rcParams['xtick.major.size'] = 1.3
rcParams['ytick.major.size'] = 1.3
rcParams['xtick.major.width'] = 0.4
rcParams['ytick.major.width'] = 0.4
rcParams['axes.linewidth'] = 0.1
rcParams['axes.edgecolor'] = COLOR
rcParams['axes.axisbelow'] = True
rcParams['xtick.color'] = COLOR
rcParams['ytick.color'] = COLOR

clrs = {
    'ADN':'#EA9E8D',
    'LMN':'#8BA6A9',
    'PSB':'#CACC90'
    }

###############################################################################################################
# LOADING DATA
###############################################################################################################
data = cPickle.load(    
    open('/home/guillaume/Dropbox/CosyneData/DATA_MODEL_RNN4.pickle', 'rb')
    )




r_psb = data['psb']
r_adn = data['adn']
r_lmn = data['lmn']
N_t =r_adn.shape[0]

epochs = np.zeros((N_t, 2))
epochs[0:N_t//4,0] = 1.0
epochs[:,1] = 1.0
epochs[N_t//2:3*N_t//4,1]  = 0.0


tmp = np.array_split(r_adn, 4)

# r_adns = [
#     tmp[0], # wake
#     np.vstack((tmp[1], tmp[3])), # sleep
#     tmp[2] # opto
# ]


r_adns = np.array_split(r_adn, 4)


hmaps = []

for r in r_adns:
    tmp = r.copy()
    # tmp -= tmp.mean(0)
    # tmp /= tmp.std(0)    
    r_adn2 = []
    step = 50
    for i in np.arange(0, tmp.shape[0], step):
        r_adn2.append(tmp[i:i+step].mean(0))
    r_adn2 = np.array(r_adn2)    

    tmp = np.hstack((r_adn2, r_adn2, r_adn2))
    tmp = gaussian_filter(tmp, (5,5))
    r_adn2 = tmp[:,r_adn2.shape[1]:r_adn2.shape[1]*2]


    rsum = r_adn2.sum(1)
    imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(r_adn2[rsum>np.median(rsum)])
    # imap = Isomap(n_components=2).fit_transform(r_adn2[rsum>np.percentile(rsum, 25)])
    bins = np.linspace(-1.1, 1.1, 30)
    # bins = np.linspace(imap.min(), imap.max(), 30)
    imap2 = np.histogram2d(imap[:,0], imap[:,1], (bins,bins))[0]
    hmaps.append(imap2)



###############################################################################################################
# PLOT
###############################################################################################################
markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(2))

outergs = gridspec.GridSpec(4, 1, figure=fig, 
    wspace = 0.5, hspace = 0.2, height_ratios = [0.3, 0.1, 0.4, 0.3])

############
# MODEL 
gs0 = subplot(outergs[0,0])
noaxis(gs0)
ylim(0, 1)
xlim(0, 1)

    
order = ['LMN', 'ADN', 'PSB']

style = ArrowStyle('Fancy', head_length=1, head_width=1.5, tail_width=0.5)


for i, (xpos, ypos) in enumerate(zip([0.3, 0.4, 0.5], [0.2, 0.5, 0.8])):
    gca().text(xpos-0.15, ypos+0.1, order[i])
    gca().add_patch(
        Ellipse((xpos, ypos), 
            0.3, 0.2, 
            linewidth=1.5,
            fill=False,
            edgecolor=clrs[order[i]]),
        )
    if i == 0:
        gca().add_patch(
            FancyArrowPatch(
                (0.3, 0.2), 
                (0.4, 0.5), 
                mutation_scale = 3,
                alpha=0.5,
                facecolor=COLOR,
                edgecolor=None,
                transform=gca().transAxes,
                arrowstyle=style        
                ))
    if i == 1:
        gca().add_patch(
            FancyArrowPatch(
                (0.4, 0.5), 
                (0.5, 0.8), 
                mutation_scale = 3,
                facecolor=COLOR,
                edgecolor=COLOR,
                alpha=0.5,
                transform=gca().transAxes,
                arrowstyle=style        
                ))

gca().add_patch(
    FancyArrowPatch(
        (0.6, 0.7), 
        (0.46, 0.2),
        mutation_scale = 3,
        facecolor=COLOR,
        edgecolor=COLOR,
        alpha=0.5,
        transform=gca().transAxes,
        arrowstyle=style,
        connectionstyle="angle3,angleA=90,angleB=20"
        ))


style = ArrowStyle('Fancy', head_length=1, head_width=1.3, tail_width=0.7)
gca().add_patch(
    FancyArrowPatch(
        (0.11, 0.2), 
        (0.06, 0.2), 
        mutation_scale = 4,
        facecolor="white",
        edgecolor=COLOR,
        transform=gca().transAxes,
        arrowstyle=style        
        ))
gca().add_patch(
    FancyArrowPatch(
        (0.21, 0.47), 
        (0.16, 0.63), 
        mutation_scale = 4,
        facecolor="white",
        edgecolor=COLOR,
        transform=gca().transAxes,
        arrowstyle=style        
        ))
gca().add_patch(
    FancyArrowPatch(
        (0.66, 0.85), 
        (0.73, 0.85), 
        mutation_scale = 4,
        facecolor="white",
        edgecolor=COLOR,
        transform=gca().transAxes,
        arrowstyle=style        
        ))


ax = gca()


# LMN LMN WEIGHTS
axin1 = ax.inset_axes([-0.1, 0.2, 0.15, 0.2])
simpleaxis(axin1)
x = np.arange(-10, 11)
y = np.exp(-(x**2)/15)
axin1.plot(x, y, 'o-', linewidth=1, markersize=1, color = clrs['LMN'])
axin1.set_ylabel(r"$W_{i-j}^{LMN}$", rotation=0, labelpad = 15, y=0.0)
axin1.set_xticks([0], ['i'])
axin1.set_yticks([])

# ADN LMN 
axin2 = ax.inset_axes([0.05, 0.7, 0.1, 0.2])
simpleaxis(axin2)
x = np.arange(0, 1, 0.01)
y = 1/(1+np.exp(-(x-0.5)*25))
axin2.plot(x, y, '-', linewidth=1, markersize=1, color = COLOR)
axin2.set_ylabel(r"$I^{ADN}$", rotation=0, labelpad = 12, y=0.0)
axin2.set_xlabel(r"$r^{LMN}$", labelpad = 4)
axin2.set_xticks([])
axin2.set_yticks([])

# PSB INH
axin3 = ax.inset_axes([0.85, 0.7, 0.1, 0.3])
noaxis(axin3)
x = np.arange(4)
axin3.plot([x[0], x[1]], [0, 1], linewidth=0.5, color = COLOR)
axin3.plot([x[3], x[0]], [0, 1], linewidth=0.5, color = COLOR)
axin3.plot([x[2], x[2]], [0, 1], linewidth=0.5, color = COLOR)
axin3.plot([x[1], x[2]], [0, 1], linewidth=0.5, color = COLOR)
axin3.plot(x, np.zeros_like(x), '^', color=clrs['PSB'], markersize=2)
axin3.plot(x, np.ones_like(x), 'o', color=clrs['PSB'], markersize=2)
axin3.set_yticks([0, 1], ['Exc.', 'Inh.'])
axin3.yaxis.set_ticks_position('none') 
axin3.set_ylim(-0.2, 1.2)


    

############/
# EPOCHS
gs1 = gridspec.GridSpecFromSubplotSpec(2,1, outergs[1,0], hspace = 0.35)

# WAKE/SLEEP
subplot(gs1[0,0])
simpleaxis(gca())
plot(epochs[:,0], linewidth = 1, color = 'grey')
# fill_between(np.arange(0, len(epochs)), np.zeros(len(epochs)), epochs[:,0], linewidth=0, color=COLOR, alpha=0.25)
xlim(0, N_t)
xticks([])
ylabel("Input", rotation=0, labelpad=15, y=0.0)
yticks([1], ["1"])


# OPTO
subplot(gs1[1,0])
simpleaxis(gca())
plot(epochs[:,1], linewidth =1, color = 'grey')
# fill_between(np.arange(0, len(epochs)), np.zeros(len(epochs)), epochs[:,1], linewidth=0, color=COLOR, alpha=0.25)
xlim(0, N_t)
xticks([])
ylabel(r"$W_{PSB-LMN}$", rotation=0, labelpad=22, y=0.0)
yticks([1], ["1"])


############
# RASTER
gs2 = gridspec.GridSpecFromSubplotSpec(3,1, outergs[2,0], hspace = 0.2)

names = ['LMN', 'ADN', 'PSB']

for i, r in enumerate([r_psb[:,0:r_adn.shape[1]], r_adn, r_lmn]):
    subplot(gs2[i,0])
    cmap = LinearSegmentedColormap.from_list(
        "", ['white', clrs[names[i]]])
    imshow(gaussian_filter(r.T, (1, 1)), 
        aspect='auto',
        cmap = cmap)
    ylabel(names[i], rotation=0, labelpad=15)
    yticks([r.shape[1]-1], [str(r.shape[1])])
    xticks([])

############
# RINGS
cmap = LinearSegmentedColormap.from_list(
         "", ["#00000000", clrs["ADN"]])
        # "", ["#00000000", "salmon"])
gs3 = gridspec.GridSpecFromSubplotSpec(1,4, outergs[3,0], hspace = 0.35)

titles = ["\"Wake\"", "\"Sleep\"", "\"Opto\"", "\"Sleep\""]



for i in range(4):
    subplot(gs3[0,i])
    noaxis(gca())

    imshow(gaussian_filter(hmaps[i], (1, 1)), 
        cmap = cmap,
        interpolation = 'bilinear')

    title(titles[i], pad=1)
    if i == 0:
        ylabel("ADN\nmanifold", labelpad = 30, rotation=0)


outergs.update(top= 0.95, bottom = 0.01, right = 0.98, left = 0.2)

#show()
savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2023 poster/figures/fig6.pdf", dpi = 100, facecolor = 'white')




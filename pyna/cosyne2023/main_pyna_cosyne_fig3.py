# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-03-02 15:21:29
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-10 15:38:36
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-02 12:42:36
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

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv

import os
import sys
sys.path.append('../')
from functions import *
sys.path.append('../../python')
import neuroseries as nts


def figsize(scale):
    fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
    #fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_width = 5
    fig_height = fig_width*golden_mean*0.9         # height in inches
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

fontsize = 9

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
rcParams['axes.linewidth'] = 0.2
rcParams['axes.edgecolor'] = COLOR
rcParams['axes.axisbelow'] = True
rcParams['xtick.color'] = COLOR
rcParams['ytick.color'] = COLOR

############################################################################################### 
# GENERAL infos
###############################################################################################
name = 'A5011-201014A'
path = '/home/guillaume/Dropbox/CosyneData/A5011-201014A'

path2 = '/home/guillaume/Dropbox/CosyneData'


############################################################################################### 
# LOADING DATA
###############################################################################################
decoding = cPickle.load(open('/home/guillaume/Dropbox/CosyneData/DATA_FIG_2_LMN_PSB.pickle', 'rb'))

angle_wak = decoding['wak']
angle_rem = decoding['rem']
angle_sws = decoding['sws']
tcurves = decoding['tcurves']
angle = decoding['angle']
peaks = decoding['peaks']
spikes = decoding['spikes']
up_ep = decoding['up_ep']
down_ep = decoding['down_ep']
tokeep = decoding['tokeep']

groups = spikes[tokeep]._metadata.groupby("location").groups
psb = groups['psb']
lmn = groups['lmn']

lmn = peaks[lmn].sort_values().index.values
psb = peaks[psb].sort_values().index.values


names = ['PSB', 'LMN']

###############################################################################################################
# PLOT
###############################################################################################################

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(2))

outergs = GridSpec(2,1, figure = fig, height_ratios = [0.3, 0.5], hspace = 0.3)

#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,2, 
    subplot_spec = outergs[0,0], width_ratios = [0.2, 0.6],
    wspace = 0.15)

names = ['PSB', 'LMN']
clrs = ['#CACC90', '#8BA6A9']


#####################################
# Histo
#####################################
gs_histo = gridspec.GridSpecFromSubplotSpec(2,1, 
    subplot_spec = gs1[0,0],
    wspace = 0.01, hspace = 0.4)


# subplot(gs_histo[0,0])
# noaxis(gca())
# img = mpimg.imread('/home/guillaume/Dropbox/CosyneData/histo_adn.png')
# imshow(img[:, :, 0], aspect='equal', cmap='viridis')
# # title("ADN")
# xticks([])
# yticks([])

# subplot(gs_histo[1,0])
# noaxis(gca())
# img = mpimg.imread('/home/guillaume/Dropbox/CosyneData/histo_lmn.png')
# imshow(img[:,:,0], aspect='equal', cmap = 'viridis')
# # title("LMN")
# xticks([])
# yticks([])


#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[0,0])
    # height_ratios=[0.5, 0.3, 0.65])

for i, st in enumerate([psb, lmn]):
    gs_tc2 = gridspec.GridSpecFromSubplotSpec(4,7, subplot_spec = gs_tc[0,i], hspace=0.4)
    idx = np.mgrid[0:4, 0:7].reshape(2, 4*7).T
    for j, n in enumerate(peaks[st].sort_values().index.values[::-1]):
        subplot(gs_tc2[idx[j,0],idx[j,1]], projection='polar')        
        tmp = tcurves[n]
        tmp = tmp/tmp.max()
        fill_between(tmp.index.values,
            np.zeros_like(tmp.index.values),
            tmp.values,            
            color = clrs[i]
            )
        xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
        yticks([])
        # xlim(0, 2*np.pi)
        # ylim(0, 1.3)
        if j == 7:
            ylabel(names[i], labelpad=20, rotation = 0, y = -0.4)
        # if j == 1:
        #     ylabel(str(len(st)), rotation = 0, labelpad = 5)

    # if i == 1: 
    #     xticks([0, 2*np.pi], [0, 360])
    #     xlabel("HD (deg.)", labelpad=-5)



#########################
# RASTER PLOTS
#########################
gs_raster = gridspec.GridSpecFromSubplotSpec(3,3, 
    subplot_spec = outergs[1,0],  hspace = 0.2)
    # height_ratios=[0.3, 0.7, 0.65, 0.5])
exs = { 'wak':nap.IntervalSet(start = 9968.5, end = 9987, time_units='s'),
        'rem':nap.IntervalSet(start = 13383.819, end= 13390, time_units = 's'),
        #'sws':nap.IntervalSet(start = 6555.6578, end = 6557.0760, time_units = 's')}
        #'sws':nap.IntervalSet(start = 5318.6593, end = 5320.0163, time_units = 's')
        'sws':nap.IntervalSet(start = 5897.10, end = 5898.45, time_units = 's'),
        # 'sws':nap.IntervalSet(start = 5895.30, end = 5898.45, time_units = 's')
        'nrem2':nap.IntervalSet(start = 5800.71, end = 5805.2, time_units = 's'),
        'nrem3':nap.IntervalSet(start = 5808.5, end = 5812.7, time_units = 's')
        }

mks = 1.8
alp = 1
medw = 0.9

# epochs = ['Wake', 'REM sleep', 'nREM sleep']
epochs = ['Wake', 'nREM sleep', 'nREM sleep']

power = cPickle.load(open('/home/guillaume/Dropbox/CosyneData/DELTA_POWER_PSB.pickle', 'rb'))

delta = power['LMN-PSB/A3019/A3019-220701A']



# for i, ep in enumerate(exs.keys()):
for i, ep in enumerate(['wak', 'nrem2', 'nrem3']):

    if ep == 'wak':
        subplot(gs_raster[0,0])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)        
        tmp = angle.restrict(exs[ep])
        tmp = tmp.as_series().rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)    
        plot(tmp, linewidth = 1, color = (0.4, 0.4, 0.4), label = 'HD')
        tmp2 = angle_wak        
        tmp2 = smoothAngle(tmp2, 1)
        tmp2 = tmp2.restrict(exs[ep])
        plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp, label = 'Decoded HD')
        title(epochs[0], pad=1)
        xticks([])
        yticks([0, 2*np.pi], ["0", "360"])
        ylim(0, 2*np.pi)
        # if j == 1:
        legend(frameon=False, handlelength = 1.5, bbox_to_anchor=(1.3,2.0), ncol = 2)


    if ep == 'rem':
        subplot(gs_raster[0,1])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        gca().spines['left'].set_visible(False)                
        tmp2 = angle_rem.restrict(exs[ep])
        plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp)
        title(epochs[1], pad = 1)
        yticks([])
        xticks([])
        ylim(0, 2*np.pi)

    if ep == 'sws':
        subplot(gs_raster[0,2])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        gca().spines['left'].set_visible(False)                        
        tmp2 = angle_sws.restrict(exs[ep])
        plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp)
        title(epochs[2], pad = 1)
        yticks([])
        xticks([])
        ylim(0, 2*np.pi)

    if ep == 'nrem2':
        subplot(gs_raster[0,1])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        gca().spines['left'].set_visible(False)                        
        tmp2 = angle_sws.restrict(exs[ep])
        plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp)
        title(epochs[1], pad = 1, x=1.1)
        yticks([])
        xticks([])
        ylim(0, 2*np.pi)

    if ep == 'nrem3':
        subplot(gs_raster[0,2])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        gca().spines['left'].set_visible(False)                        
        tmp2 = angle_sws.restrict(exs[ep])
        plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp)
        # title(epochs[2], pad = 1)
        yticks([])
        xticks([])
        ylim(0, 2*np.pi)

    for j, st in enumerate([psb, lmn]):
        subplot(gs_raster[j+1,i])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        if i > 0: gca().spines['left'].set_visible(False)

        order = tcurves[st].idxmax().sort_values().index.values

        for k, n in enumerate(order):
            spk = spikes[n].restrict(exs[ep]).index.values
            if len(spk):
                clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,90,65])
                clr = clrs[j]
                #clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.7])
                # plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.5)
                plot(spk, np.ones_like(spk)*k, '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.5)

        # ylim(0, 2*np.pi)
        xlim(exs[ep].loc[0,'start'], exs[ep].loc[0,'end'])
        xticks([])
        if i == 0:
            yticks([len(st)-1], [len(st)])
        else:
            yticks([])
        gca().spines['bottom'].set_visible(False)

        

        if i == 0 and j == 1:           
            plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('1s', horizontalalignment='right', x=1.0)
        # if i == 1 and j == 1:           
        #     plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
        #     xlabel('1s', horizontalalignment='right', x=1.0)
        if i in [1,2] and j == 1:
            plot(np.array([exs[ep].end[0]-0.5, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('0.5s', horizontalalignment='right', x=1.0)


outergs.update(top= 0.97, bottom = 0.05, right = 0.96, left = 0.1)


savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2023 poster/figures/fig3.pdf", dpi = 200, facecolor = 'white')

#show() 
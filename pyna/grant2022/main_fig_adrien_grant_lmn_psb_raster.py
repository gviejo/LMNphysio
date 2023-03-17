# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-02-20 12:27:31
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-20 18:47:33
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-10 16:57:11
import numpy as np
import pandas as pd
import pynapple as nap

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
    fig_width = 7
    fig_height = fig_width*golden_mean*1         # height in inches
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
rcParams['axes.linewidth'] = 0.2
rcParams['axes.edgecolor'] = COLOR
rcParams['axes.axisbelow'] = True
rcParams['xtick.color'] = COLOR
rcParams['ytick.color'] = COLOR

############################################################################################### 
# GENERAL infos
###############################################################################################
name = 'A3019-220701A'
path = '/mnt/DataRAID2/LMN-PSB/A3019/A3019-220701A'

path2 = '/home/guillaume/Dropbox/CosyneData'



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

outergs = GridSpec(2,1, figure = fig, height_ratios = [0.6, 0.3], hspace = 0.45)

#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,3, 
    subplot_spec = outergs[0,0], width_ratios = [0.05, 0.2, 0.5],
    wspace = 0.15)

names = ['PSB', 'LMN']
clrs = ['#EA9E8D', '#8BA6A9']


#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = gs1[0,1])

for i, st in enumerate([psb, lmn]):

    # subplot(gs_tc[i+1, 0])
    # order = peaks[st].sort_values().index.values[::-1]
    # tmp = tcurves[order].values.T
    # tmp = tmp/tmp.max(0)
    # imshow(tmp, aspect='auto')

    
    gs_tc2 = gridspec.GridSpecFromSubplotSpec(len(st),1, subplot_spec = gs_tc[i+1,0], hspace=0.4)
    for j, n in enumerate(peaks[st].sort_values().index.values[::-1]):
        subplot(gs_tc2[j,0])
        simpleaxis(gca())       
        clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,90,65])
        # clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
        tmp = tcurves[n]
        tmp = tmp/tmp.max()
        fill_between(tmp.index.values,
            np.zeros_like(tmp.index.values),
            tmp.values,
            # color = clr
            color = clrs[i]
            )
        xticks([])
        yticks([])
        xlim(0, 2*np.pi)
        ylim(0, 1.3)
        if j == (len(st)//2)+2:
            ylabel(names[i], labelpad=15, rotation = 0)
        if j == 1:
            ylabel(str(len(st)), rotation = 0, labelpad = 5)

    if i == 1: 
        xticks([0, 2*np.pi], [0, 360])
        xlabel("HD (deg.)", labelpad=-5)


#########################
# RASTER PLOTS
#########################
gs_raster = gridspec.GridSpecFromSubplotSpec(3,3, subplot_spec = gs1[0,2],  hspace = 0.2)
exs = { 'wak':nap.IntervalSet(start = 9968.5, end = 9987, time_units='s'),
        'rem':nap.IntervalSet(start = 13383.819, end= 13390, time_units = 's'),
        #'sws':nap.IntervalSet(start = 6555.6578, end = 6557.0760, time_units = 's')}
        #'sws':nap.IntervalSet(start = 5318.6593, end = 5320.0163, time_units = 's')
        'sws':nap.IntervalSet(start = 5896.30, end = 5898.45, time_units = 's')
        }

mks = 2
alp = 1
medw = 0.8

epochs = ['Wake', 'REM sleep', 'nREM sleep']


for i, ep in enumerate(exs.keys()):

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
        legend(frameon=False, handlelength = 1.5, bbox_to_anchor=(-0.1,1.25))


    if ep == 'rem':
        subplot(gs_raster[0,1])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        gca().spines['left'].set_visible(False)                
        tmp2 = angle_rem.restrict(exs[ep])
        plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp)
        ylim(0, 2*np.pi)
        title(epochs[1], pad = 1)
        yticks([])
        xticks([])

    if ep == 'sws':
        subplot(gs_raster[0,2])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        gca().spines['left'].set_visible(False)                        
        tmp2 = angle_sws.restrict(exs[ep])
        plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp)
        title(epochs[2], pad = 1)
        ylim(0, 2*np.pi)
        yticks([])
        xticks([])

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
        yticks([])
        gca().spines['bottom'].set_visible(False)

        

        if i == 0 and j == 1:           
            plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('1s', horizontalalignment='right', x=1.0)
        if i == 1 and j == 1:           
            plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('1s', horizontalalignment='right', x=1.0)
        if i == 2 and j == 1:           
            plot(np.array([exs[ep].end[0]-0.2, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('0.2s', horizontalalignment='right', x=1.0)


#####################################################################
# OPTO
#####################################################################

path = '/mnt/Data2/Opto/A8000/A8044/A8044-230217A'
#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220614A'

data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('rate', 0.6)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 40, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

opto_sleep_ep = loadOptoEp(path, epoch=1, n_channels = 2, channel = 0)

opto_wake_ep = loadOptoEp(path, epoch=3, n_channels = 2, channel = 0)


# stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

# peth = nap.compute_perievent(spikes, nap.Ts(opto_ep["start"].values), minmax=(-stim_duration, 2*stim_duration))

# frates = pd.DataFrame({n:peth[n].count(0.05).sum(1) for n in peth.keys()})

# rasters = {j:pd.concat([peth[j][i].as_series().fillna(i) for i in peth[j].index]) for j in peth.keys()}


tcurves = tuning_curves
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

hd = SI[SI>0.05].dropna().index.values



#####################################
gs2 = gridspec.GridSpecFromSubplotSpec(2,4, 
    subplot_spec = outergs[1,0], width_ratios = [0.1, 0.3, 0.5, 0.5],
    height_ratios = [0.25, 0.75],
    wspace = 0.15)

names = ['PSB', 'LMN']
clrs = ['#EA9E8D', '#8BA6A9']


#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(len(hd),1, subplot_spec = gs2[1,1])

for j, n in enumerate(peaks[hd].sort_values().index.values[::-1]):
    subplot(gs_tc[j,0])
    simpleaxis(gca())       
    clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,90,65])
    # clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
    tmp = tcurves[n]
    tmp = tmp/tmp.max()
    fill_between(tmp.index.values,
        np.zeros_like(tmp.index.values),
        tmp.values,
        # color = clr
        color = clrs[0]
        )
    xticks([])
    yticks([])
    xlim(0, 2*np.pi)
    ylim(0, 1.3)
    if j == (len(st)//2)+2:
        ylabel(names[0], labelpad=15, rotation = 0)
    if j == 1:
        ylabel(str(len(hd)), rotation = 0, labelpad = 5)

if i == 1: 
    xticks([0, 2*np.pi], [0, 360])
    xlabel("HD (deg.)", labelpad=-5)


# #########################
# # RASTER PLOTS
# #########################


mks = 2
alp = 1
medw = 0.8


for i, name, opto_ep in zip(range(2), ['Wake', 'Sleep'], [opto_wake_ep, opto_sleep_ep]):

    stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

    eps = opto_ep.loc[0:2]

    start = eps.start.values[0] - stim_duration*2
    end = eps.end.values[-1] + stim_duration*2

    exep = nap.IntervalSet(start=start-stim_duration, end = end+stim_duration)

    mua = spikes[hd].restrict(exep).count(0.1).sum(1)

    # Mua count
    subplot(gs2[0,i+2])
    simpleaxis(gca())
    gca().spines['bottom'].set_visible(False)
    xticks([])
    xlim(start, end)
    [axvspan(s, e, color = 'red', alpha = 0.1, linewidth =0) for s, e in eps.values]
    gca().spines['bottom'].set_visible(False)
    fill_between(
        mua.index.values, np.zeros_like(mua.values), 
        mua.values, color = 'darkgray', linewidth = 0)

    title(name)
    if i == 0:
        ylabel("MUA\n(count)")

    # rasters
    subplot(gs2[1,i+2])
    simpleaxis(gca())
    gca().spines['bottom'].set_visible(False)

    xlim(start, end)
    order = tcurves[hd].idxmax().sort_values().index.values

    for k, n in enumerate(order):
        spk = spikes[n].restrict(exep).index.values
        if len(spk):
            clr = clrs[0]
            plot(spk, np.ones_like(spk)*k, '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.5)

    [axvspan(s, e, color = 'red', alpha = 0.1) for s, e in eps.values]
    # ylim(0, 2*np.pi)
    # xlim(exs[ep].loc[0,'start'], exs[ep].loc[0,'end'])
    xticks([])
    yticks([])
    gca().spines['bottom'].set_visible(False)



outergs.update(top= 0.97, bottom = 0.08, right = 0.96, left = 0.025)


savefig("/home/guillaume/LMNphysio/figures/figures_adrien_2022/fig_lmn_psb.pdf", dpi = 200, facecolor = 'white')
#show() 
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-03-02 15:21:29
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-10 18:21:05
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
    fig_width = 6
    fig_height = fig_width*golden_mean*0.5         # height in inches
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
# LOADING DATA
###############################################################################################
path = '/mnt/Data2/Opto/A8000/A8047/A8047-230310A'
#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220614A'

data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('rate', 0.6)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals("sws")

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

spikes = spikes.getby_category("location")["lmn"].getby_threshold('SI', 0.1)

lmn = spikes.index

tcurves = tuning_curves
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

opto_ep = loadOptoEp(path, epoch=1, n_channels = 2, channel = 0)



###############################################################################################################
# PLOT
###############################################################################################################

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(2))

outergs = GridSpec(2,1, figure = fig, height_ratios = [0.5, 0.5], hspace = 0.3)

#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,3, 
    subplot_spec = outergs[0,0], width_ratios = [0.4,0.1, 0.6],
    wspace = 0.3)

names = ['PSB', 'LMN']
clrs = ['#CACC90', '#8BA6A9']


#####################################
# Histo
#####################################

subplot(gs1[0,0])
noaxis(gca())
img = mpimg.imread('/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2023 poster/figures/LMNopto.png')
imshow(img, aspect='equal')


#########################
# TUNING CURVes
#########################
gs_tc2 = gridspec.GridSpecFromSubplotSpec(4,2, subplot_spec = gs1[0,1], hspace=0.4)
idx = np.mgrid[0:4, 0:2].reshape(2, 4*2).T
for j, n in enumerate(peaks[lmn].sort_values().index.values[::-1]):
    subplot(gs_tc2[idx[j,0],idx[j,1]], projection='polar')        
    tmp = tcurves[n]
    tmp = tmp/tmp.max()
    fill_between(tmp.index.values,
        np.zeros_like(tmp.index.values),
        tmp.values,            
        color = clrs[1]
        )
    xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
    yticks([])
    # xlim(0, 2*np.pi)
    # ylim(0, 1.3)
    if j == 7:
        ylabel(names[1], labelpad=20, rotation = 0, y = -0.4)
    # if j == 1:
    #     ylabel(str(len(st)), rotation = 0, labelpad = 5)

# if i == 1: 
#     xticks([0, 2*np.pi], [0, 360])
#     xlabel("HD (deg.)", labelpad=-5)



#########################
# RASTER PLOTS
#########################


mks = 3.0
alp = 1
medw = 1.5

# exs = nap.IntervalSet(start = 4191, end = 4197)
exs = nap.IntervalSet(start = 4191, end = 4197)

exopto_ep = exs.intersect(opto_ep)

gs_raster = gridspec.GridSpecFromSubplotSpec(3,1, 
    subplot_spec = gs1[0,2], hspace=0.2,
    height_ratios=[0.1, 0.3, 0.1]
    )
subplot(gs_raster[1,0])
simpleaxis(gca())

order = peaks[lmn].sort_values().index.values

for k, n in enumerate(order):
    spk = spikes[n].restrict(exs).index.values
    if len(spk):
        clr = clrs[1]
        plot(spk, np.ones_like(spk)*k, '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.8)

axvspan(exopto_ep.loc[0,'start'], exopto_ep.loc[0,'end'], 
    color = 'lightcoral', alpha=0.25,
    linewidth =0
    )
yticks([len(lmn)-1], [str(len(lmn))])
xlim(exs.loc[0,'start'], exs.loc[0,'end'])
xticks([])
xlabel(str(int(exs.tot_length('s')))+' s', horizontalalignment='right', x=1.0)
ylabel("LMN", rotation=0, labelpad=8, y=0.3)
title("Optogenetic inactivation of PSB")


##########################################################
# MUA activity
##########################################################

dataopto = cPickle.load(open('/home/guillaume/Dropbox/CosyneData/OPTO_SLEEP_A8047.pickle', 'rb'))


frates = dataopto['frates']



gs2 = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec = outergs[1,0], wspace = 0.5)
    # width_ratios = [0.3, 0.25], wspace = 0.15)#, hspace = 0.5)



subplot(gs2[0,0])
simpleaxis(gca())

fr = frates/frates.loc[1:].mean()
fr = fr.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=5.0)

# plot(fr, linewidth = 1, alpha = 0.5, color = clrs[1])
plot(fr.mean(1), linewidth = 1.5, alpha=1.0, color = clrs[1])
m = fr.mean(1).values
x = fr.index.values
s = fr.sem(1).values
fill_between(x, m-s, m+s, color=clrs[1], alpha=0.25, linewidth=0.0)
ylim(0, fr.max().max())
yticks([0, 1], [0, 100])
xticks([0, 1], [0, 1])

axvspan(0, 1, 
    color = 'lightcoral', alpha=0.25,
    linewidth =0
    )

xlabel("Time from light onset (s)")
ylabel("Rate\nmod.\n(%)", rotation=0, y = 0.3, labelpad = 15)

##########################################################
# PEARSON R
##########################################################

gs22 = gridspec.GridSpecFromSubplotSpec(1,3, 
    subplot_spec = gs2[0,1:], width_ratios = [0.4,0.4, 0.4],
    wspace = 0.5)

r = dataopto['r']
mkrs = 3
subplot(gs22[0,0])
simpleaxis(gca())
plot(r['wak'], r['opto'], 'o', color = 'lightcoral', alpha = 0.5, markersize = mkrs, markeredgewidth=0)
m, b = np.polyfit(r['wak'].values, r['opto'].values, 1)
x = np.linspace(r['wak'].min(), r['wak'].max(),5)
plot(x, x*m + b, label = 'Opto.\nr='+str(np.round(m,2)), color = 'lightcoral', linewidth=1)
xlabel('Wake corr (r)')
ylabel('corr (r)', labelpad=10, rotation=0)
plot(r['wak'], r['nopto'], 'o', color = 'grey', alpha = 0.5, markeredgewidth=0, markersize=mkrs)
m, b = np.polyfit(r['wak'].values, r['nopto'].values, 1)
x = np.linspace(r['wak'].min(), r['wak'].max(),5)
plot(x, x*m + b, label = 'NREM\nr='+str(np.round(m,2)), color = 'grey', linewidth=1)
legend(frameon=False, handlelength = 0.5, bbox_to_anchor=(1.0,1.0))#, ncol = 2)

subplot(gs22[0,2])
simpleaxis(gca())

for p in r.index:
    plot([0,1],r.loc[p,['nopto', 'opto']], 'o-', color = clrs[1], markersize = mkrs, linewidth=1)
xticks([0,1], ['NREM\nsleep', 'Opto.\nsleep'])
ylabel("Pairwise\nCorr. (r)", rotation=0, labelpad=12, y=0.4)
xlim(-0.5, 1.5)




    

outergs.update(top= 0.97, bottom = 0.2, right = 0.96, left = 0.12)


savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2023 poster/figures/fig4.pdf", dpi = 200, facecolor = 'white')

#show() 
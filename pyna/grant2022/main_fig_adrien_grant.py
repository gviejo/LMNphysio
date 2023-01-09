# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-08-10 16:37:50
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-09-02 17:21:29

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
    fig_height = fig_width*golden_mean*1.9         # height in inches
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

rcParams['font.family'] = 'Helvetica'
rcParams['font.size'] = fontsize
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
rcParams['axes.linewidth'] = 0.6
rcParams['axes.edgecolor'] = 'grey'
rcParams['axes.axisbelow'] = True
# rcParams['xtick.color'] = 'grey'
# rcParams['ytick.color'] = 'grey'

############################################################################################### 
# GENERAL infos
###############################################################################################
name = 'A5011-201014A'
path = '/home/guillaume/Dropbox/CosyneData/A5011-201014A'

path2 = '/home/guillaume/Dropbox/CosyneData'


############################################################################################### 
# LOADING DATA
###############################################################################################
data = nap.load_session(path, 'neurosuite')

spikes = data.spikes#.getby_threshold('freq', 1.0)
angle = data.position['ry']
position = data.position
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')
rem_ep = data.read_neuroscope_intervals('rem')

# Only taking the first wake ep
wake_ep = wake_ep.loc[[0]]

adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
spikes = spikes.getby_threshold('SI', 0.1, op = '>')
tuning_curves = tuning_curves[spikes.keys()]

tokeep = list(spikes.keys())

adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values

tcurves = tuning_curves

tokeep = np.hstack((adn, lmn))

tmp = cPickle.load(open(path2+'/figures_poster_2021/fig_cosyne_decoding.pickle', 'rb'))

decoding = {
    'wak':nap.Tsd(t=tmp['wak'].index.values, d=tmp['wak'].values, time_units = 'us'),
    'sws':nap.Tsd(t=tmp['sws'].index.values, d=tmp['sws'].values, time_units = 'us'),
    'rem':nap.Tsd(t=tmp['rem'].index.values, d=tmp['rem'].values, time_units = 'us'),   
}


tmp = cPickle.load(open(path2+'/figures_poster_2022/fig_cosyne_decoding.pickle', 'rb'))
peaks = tmp['peaks']

###############################################################################################################
# PLOT
###############################################################################################################

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(3))

outergs = GridSpec(3,1, figure = fig, height_ratios = [0.3, 0.3, 0.5], hspace = 0.4)

########################################################################################################
########################################################################################################
# LMN - ADN
########################################################################################################
########################################################################################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[0,0], width_ratios = [0.1, 0.5])

names = ['ADN', 'LMN']

#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gs1[0,0])

for i, st in enumerate([adn, lmn]):
    gs_tc2 = gridspec.GridSpecFromSubplotSpec(len(st),1, subplot_spec = gs_tc[i,0])
    for j, n in enumerate(peaks[st].sort_values().index.values[::-1]):
        subplot(gs_tc2[j,0])
        simpleaxis(gca())       
        #clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
        clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
        fill_between(tcurves[n].index.values,
            np.zeros_like(tcurves[n].index.values),
            tcurves[n].values,
            color = clr
            )
        xticks([])
        yticks([])
        xlim(0, 2*np.pi)
        if j == len(st)//2:
            ylabel(names[i], labelpad=20, rotation = 0)
        if j == 0:
            ylabel(str(len(st)), rotation = 0, labelpad = 8)

    if i == 1: xticks([0, 2*np.pi], [0, 360])


#########################
# RASTER PLOTS
#########################
gs_raster = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec = gs1[0,1],  hspace = 0.2)
exs = { 'wak':nap.IntervalSet(start = 7590.0, end = 7600.0, time_units='s'),
        'rem':nap.IntervalSet(start = 15710.150000, end= 15720.363258, time_units = 's'),
        'sws':nap.IntervalSet(start = 4400600.000, end = 4402154.216186978, time_units = 'ms')}

mks = 2
alp = 1
medw = 0.8

epochs = ['Wake', 'REM sleep', 'nREM sleep']

for i, ep in enumerate(exs.keys()):
    for j, st in enumerate([adn, lmn]):
        subplot(gs_raster[j,i])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        if i > 0: gca().spines['left'].set_visible(False)
        for k, n in enumerate(st):
            spk = spikes[n].restrict(exs[ep]).index.values
            if len(spk):
                #clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
                clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
                plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.5)
        
        ylim(0, 2*np.pi)
        xlim(exs[ep].loc[0,'start'], exs[ep].loc[0,'end'])
        xticks([])
        gca().spines['bottom'].set_visible(False)

        if i == 0: 
            yticks([0, 2*np.pi], ["0", "360"])          
        else:
            yticks([])
        
        if ep == 'wak':
            tmp = position['ry'].restrict(exs[ep])
            tmp = tmp.as_series().rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)    
            plot(tmp, linewidth = 1, color = 'black', label = 'Head-direction')
            tmp2 = decoding['wak']
            tmp2 = nap.Tsd(tmp2, time_support = wake_ep)
            tmp2 = smoothAngle(tmp2, 1)
            tmp2 = tmp2.restrict(exs[ep])
            plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp) 
            if j == 1:
                legend(frameon=False, handlelength = 1, bbox_to_anchor=(1,-0.1))

        if ep == 'rem':
            tmp2 = decoding['rem'].restrict(exs[ep])
            plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp, label = 'Decoded head-direction')
            if j == 1:
                legend(frameon=False, handlelength = 2, bbox_to_anchor=(1.5,-0.1))

        if ep == 'sws':
            tmp2 = decoding['sws']
            tmp3 = pd.Series(index = tmp2.index, data = np.unwrap(tmp2.values)).rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
            tmp3 = tmp3%(2*np.pi)
            tmp2 = nap.Tsd(tmp3).restrict(exs[ep])
            plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 1, color = 'gray', alpha = alp)
            plot(tmp2.loc[tmp2.idxmax()+0.03:],'--', linewidth = 1, color = 'gray', alpha = alp)


        if i == 0 and j == 1:           
            plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('1s', horizontalalignment='right', x=1.0)
        if i == 1 and j == 1:           
            plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('1s', horizontalalignment='right', x=1.0)
        if i == 2 and j == 1:           
            plot(np.array([exs[ep].end[0]-0.5, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('0.5s', horizontalalignment='right', x=1.0)

        if j == 0:
            title(epochs[i], pad = -1)






########################################################################################################
########################################################################################################
# LMN - PSB
########################################################################################################
########################################################################################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[1,0], width_ratios = [0.1, 0.5])



import _pickle as cPickle
decoding = cPickle.load(open('../../figures/figures_adrien_2022/fig_1_decoding.pickle', 'rb'))

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

groups = spikes._metadata.groupby("location").groups
psb = groups['psb']
lmn = groups['lmn']

lmn = peaks[lmn].sort_values().index.values
psb = peaks[psb].sort_values().index.values

names = ['PSB', 'LMN']


#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gs1[0,0])

for i, st in enumerate([psb, lmn]):
    st = np.intersect1d(st, tokeep)
    gs_tc2 = gridspec.GridSpecFromSubplotSpec(len(st),1, subplot_spec = gs_tc[i,0])    
    for j, n in enumerate(peaks[st].sort_values().index.values[::-1]):        
        subplot(gs_tc2[j,0])
        simpleaxis(gca())       
        #clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
        clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
        fill_between(tcurves[n].index.values,
            np.zeros_like(tcurves[n].index.values),
            tcurves[n].values,
            color = clr
            )
        xticks([])
        yticks([])
        xlim(0, 2*np.pi)
        if j == len(st)//2:
            ylabel(names[i], labelpad=20, rotation = 0)
        if j == 0:
            ylabel(str(len(st)), rotation = 0, labelpad = 8)

    if i == 1: xticks([0, 2*np.pi], [0, 360])


#########################
# RASTER PLOTS
#########################
gs_raster = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec = gs1[0,1],  hspace = 0.2)
exs = { 'wak':nap.IntervalSet(start = 9910, end = 9931, time_units='s'),
        'rem':nap.IntervalSet(start = 13383.819, end= 13390, time_units = 's'),
        #'sws':nap.IntervalSet(start = 6555.6578, end = 6557.0760, time_units = 's')}
        #'sws':nap.IntervalSet(start = 5318.6593, end = 5320.0163, time_units = 's')
        'sws':nap.IntervalSet(start = 5407.4, end = 5409.25, time_units = 's')
        }


mks = 2.8
alp = 1
medw = 1
lwd = 1.5

epochs = ['Wake', 'REM sleep', 'nREM sleep']

nonhd = list(set(list(psb) + list(lmn)) - set(tokeep))


for i, ep in enumerate(exs.keys()):
    for j, st in enumerate([psb, lmn]):                
        subplot(gs_raster[j,i])
        simpleaxis(gca())
        gca().spines['bottom'].set_visible(False)
        if i > 0: gca().spines['left'].set_visible(False)
        for k, n in enumerate(st):
            spk = spikes[n].restrict(exs[ep]).index.values
            if len(spk):
                if n in tokeep:
                    clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
                    plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 1)
                elif n not in tokeep and ep=='sws':
                    plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = 'grey', markersize = mks, markeredgewidth = medw, alpha = 0.2)
        
        ylim(0, 2*np.pi)
        xlim(exs[ep].loc[0,'start'], exs[ep].loc[0,'end'])
        xticks([])
        gca().spines['bottom'].set_visible(False)

        if i == 0: 
            yticks([0, 2*np.pi], ["0", "360"])          
        else:
            yticks([])
        
        if ep == 'wak':
            tmp = angle.restrict(exs[ep])
            tmp = smoothAngle(tmp, 1)            
            plot(tmp, linewidth = 1, color = 'black', label = 'Head-direction')
            tmp2 = angle_wak
            tmp2 = smoothAngle(tmp2, 2)
            tmp2 = tmp2.restrict(exs[ep])
            plot(tmp2, '--', linewidth = lwd, color = 'darkgray', alpha = alp) 
            if j == 1:
                legend(frameon=False, handlelength = 1, bbox_to_anchor=(1,-0.1))

        if ep == 'rem':
            tmp2 = angle_rem.restrict(exs[ep])
            tmp2 = smoothAngle(tmp2, 2)
            plot(tmp2, linewidth = lwd, color = 'darkgray', alpha = alp, label = 'Decoded head-direction')
            if j == 1:
                legend(frameon=False, handlelength = 2, bbox_to_anchor=(1.5,-0.1))

        if ep == 'sws':
            tmp2 = angle_sws.restrict(exs[ep])
            tmp2 = smoothAngle(tmp2, 2)
            up_ex = up_ep.intersect(exs[ep])
            up_ex = up_ex.merge_close_intervals(20, time_units = 'ms')
            #for k in up_ex.index.values:
            down_ep = exs['sws'].set_diff(up_ex.loc[[0,2,3]])
            for k in [0, 2, 3]:
                plot(tmp2.restrict(up_ex.loc[[k]]), linewidth = lwd, color = 'darkgray', alpha = alp, label = 'Decoded head-direction')
            for s, e in down_ep.values:
                axvspan(s, e, color = 'blue', alpha=0.1)


        if i == 0 and j == 1:           
            plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('1s', horizontalalignment='right', x=1.0)
        if i == 1 and j == 1:           
            plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('1s', horizontalalignment='right', x=1.0)
        if i == 2 and j == 1:           
            plot(np.array([exs[ep].end[0]-0.2, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
            xlabel('0.2s', horizontalalignment='right', x=1.0)

        if j == 0:
            title(epochs[i], pad = -1)














#######################################################################################################
#######################################################################################################
# Correlation
#######################################################################################################
#######################################################################################################
gscor = gridspec.GridSpecFromSubplotSpec(2,4, subplot_spec = outergs[2,0], hspace = 0.4)

allr = cPickle.load(open(os.path.join('../../data/', 'All_correlation_ADN_LMN_PSB.pickle'), 'rb'))
allr = allr['allr']

allaxis=[]

clrs = ['steelblue', 'gray', 'darkorange']

mkrs = 6

eps = ['REM corr. (r)', 'NREM corr. (r)']

titles = ['LMN', 'ADN', 'PSB']

for i,st in enumerate(['lmn', 'adn', 'psb']):
    for j,ep in enumerate(['rem', 'sws']):
        subplot(gscor[j,i])
        simpleaxis(gca())
        scatter(allr[st]['wak'], allr[st][ep], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0, s = mkrs)
        m, b = np.polyfit(allr[st]['wak'].values, allr[st][ep].values, 1)
        x = np.linspace(allr[st]['wak'].min(), allr[st]['wak'].max(),5)
        r, p = scipy.stats.pearsonr(allr[st]['wak'], allr[st][ep])
        plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)), linewidth = 1)
        xlabel('Wake corr. (r)')
        ylabel(eps[j])
        legend(handlelength = 0.4, loc='center', bbox_to_anchor=(0.2, 0.7, 0.5, 0.5), framealpha =0)
        ax = gca()
        aspectratio=1.0
        ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
        #ax.set_aspect(ratio_default*aspectratio)
        ax.set_aspect(1)
        locator_params(axis='y', nbins=3)
        locator_params(axis='x', nbins=3)
        #xlabel('Wake corr. (r)')
        if j == 0: title(titles[i])
        allaxis.append(gca())


st = 'lmn'
tmpr = allr[st].dropna()
eps = ['UP corr. (r)', 'DOWN corr. (r)']
for j,ep in enumerate(['up', 'down']):
    subplot(gscor[j,-1])
    simpleaxis(gca())
    scatter(tmpr['wak'], tmpr[ep], color = clrs[0], alpha = 0.5, edgecolor = None, linewidths=0, s = mkrs)
    m, b = np.polyfit(tmpr['wak'].values, tmpr[ep].values, 1)
    x = np.linspace(tmpr['wak'].min(), tmpr['wak'].max(),5)
    r, p = scipy.stats.pearsonr(tmpr['wak'], tmpr[ep])
    plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)), linewidth = 1)
    xlabel('Wake corr. (r)')
    ylabel(eps[j])
    legend(handlelength = 0.4, loc='center', bbox_to_anchor=(0.2, 0.7, 0.5, 0.5), framealpha =0)
    ax = gca()
    aspectratio=1.0
    ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
    #ax.set_aspect(ratio_default*aspectratio)
    ax.set_aspect(1)
    locator_params(axis='y', nbins=3)
    locator_params(axis='x', nbins=3)
    #xlabel('Wake corr. (r)')
    allaxis.append(gca())

            
xlims = []
ylims = []
for ax in allaxis:
    xlims.append(ax.get_xlim())
    ylims.append(ax.get_ylim())
xlims = np.array(xlims)
ylims = np.array(ylims)
xl = (np.min(xlims[:,0]), np.max(xlims[:,1]))
yl = (np.min(ylims[:,0]), np.max(ylims[:,1]))
for ax in allaxis:
    ax.set_xlim(xl)
    ax.set_ylim(yl)
# outergs.update(top= 0.97, bottom = 0.06, right = 0.96, left = 0.06)


outergs.update(top= 0.97, bottom = 0.06, right = 0.96, left = 0.06)    

savefig("/home/guillaume/LMNphysio/figures/figures_adrien_2022/fig1.pdf", dpi = 200, facecolor = 'white')







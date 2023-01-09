# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-12-16 14:24:56
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-19 14:00:54
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
from functions import *
import sys
from itertools import combinations, product
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.pyplot import *

# import seaborn as sns
# sns.set_theme()


path = '/mnt/Data2/Opto/A8000/A8036/A8036-221219'
#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220614A'

data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('rate', 0.6)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

opto_ep = loadOptoEp(path, epoch=1, n_channels = 2, channel = 0)

opto_ep = opto_ep.merge_close_intervals(0.03)

stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

peth = nap.compute_perievent(spikes, nap.Ts(opto_ep["start"].values), minmax=(-stim_duration, 2*stim_duration))

frates = pd.DataFrame({n:peth[n].count(0.05).sum(1) for n in peth.keys()})

rasters = {j:pd.concat([peth[j][i].as_series().fillna(i) for i in peth[j].index]) for j in peth.keys()}

############################################################################################### 
# PLOT
###############################################################################################
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = spikes._metadata.group.values

figure()
count = 1
for l,j in enumerate(np.unique(shank)):
    neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
    for k,i in enumerate(neurons):      
        subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')    
        plot(tuning_curves[i], label = str(shank[l]) + ' ' + str(i), color = colors[l])
        if SI.loc[i,'SI'] > 0.1:
            fill_between(tuning_curves[i].index.values, np.zeros_like(tuning_curves[i].values), tuning_curves[i].values)
        legend()

        count+=1
        gca().set_xticklabels([])

hd = SI[SI>0.1].dropna().index.values
nhd = SI[SI<0.1].dropna().index.values

# groups = np.array_split(list(spikes.keys()), 3)
groups = [hd, nhd]
for i, neurons in enumerate(groups):
    fig = figure()
    count = 1
    m = int(np.sqrt(len(neurons)))+1
    gs0 = GridSpec(m,m, figure=fig)
    for k,n in enumerate(neurons):
        print(k//n, k%n)
        subgs = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[k//m, k%m])
        subplot(subgs[:,0], projection = 'polar')
        plot(tuning_curves[n])
        xticks([])
        yticks([])  
        subplot(subgs[0,1])     
        bar(frates[n].index.values, frates[n].values, np.diff(frates[n].index.values)[0])
        axvline(0)
        axvline(stim_duration)
        title(n+2)
        yticks([])
        subplot(subgs[1,1])
        plot(rasters[n], '.', markersize = 0.24)
        title(n+2)
        count+=1
        gca().set_xticklabels([])
        axvline(0)
        axvline(stim_duration)
        yticks([])

figure()
subplot(121)
plot(frates[hd].mean(1))
subplot(122)
plot(frates[nhd].mean(1))

show()
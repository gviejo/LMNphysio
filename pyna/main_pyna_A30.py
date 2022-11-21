# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 11:23:07
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-11-18 17:47:30
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
from functions import *
import sys
from itertools import combinations, product
from umap import UMAP
from matplotlib.pyplot import *
from sklearn.manifold import Isomap
import seaborn as sns
sns.set_theme()


path = '/mnt/DataRAID2/LMN-PSB/A3020/A3020-220723A'
#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220614A'

data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('rate', 1.0)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']

# sws_ep = data.read_neuroscope_intervals("sws")

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

pf, bins = nap.compute_2d_tuning_curves(spikes, data.position[['x', 'z']], 15, ep=wake_ep)





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
        legend()
        count+=1
        gca().set_xticklabels([])

show()


# figure()
# count = 1
# for l,j in enumerate(np.unique(shank)):
#     neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
#     for k,i in enumerate(neurons):      
#         subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count)  
#         imshow(pf[i], aspect = 'auto')
#         title(str(shank[l]) + ' ' + str(i))        
#         count+=1
#         gca().set_xticklabels([])

# show()

################
# Rasters
################
spikes = spikes.getby_threshold("SI", 0.1)

lmn = list(spikes._metadata.groupby("location").groups['lmn'])
psb = list(spikes._metadata.groupby("location").groups['psb'])


peaks = tuning_curves.idxmax()

hd = np.sort(list(psb)+list(lmn))

# decoded_sleep, proba_angle_sleep = nap.decode_1d(tuning_curves=tuning_curves[hd],
#                                                  group=spikes, 
#                                                  ep=sws_ep,
#                                                  bin_size=0.1, # second
#                                                  feature=data.position['ry'], 
#                                                  )

figure()
ax = subplot(211)
for n in psb:
    plot(spikes[n].restrict(sws_ep).restrict(sleep_ep.loc[[0]]).fillna(peaks[n]), '|')
    #plot(decoded_sleep.restrict(sleep_ep.loc[[0]]))
    ylim(0,2*np.pi)
subplot(212, sharex = ax)
for n in lmn:
    plot(spikes[n].restrict(sws_ep).restrict(sleep_ep.loc[[0]]).fillna(peaks[n]), '|')
    #plot(decoded_sleep.restrict(sleep_ep.loc[[0]]))
    ylim(0,2*np.pi)
show()



################
# Cross correlograms
################



cc_wake = nap.compute_crosscorrelogram(spikes, 0.2, 3, ep=wake_ep.loc[[0]])
cc_sws = nap.compute_crosscorrelogram(spikes, 0.002, 0.1, ep=sws_ep)

lmn = tuning_curves.idxmax()[lmn].sort_values().index
psb = tuning_curves.idxmax()[psb].sort_values().index


figure()
gs = GridSpec(len(lmn)+1, len(psb)+1)
for i, n in enumerate(lmn):
    subplot(gs[i+1,0], projection='polar')
    plot(tuning_curves[n])
    xticks([])
    yticks([])
for j, m in enumerate(psb):
    subplot(gs[0,j+1], projection='polar')
    plot(tuning_curves[m])
    xticks([])
    yticks([])

for i,n in enumerate(lmn):
    for j,m in enumerate(psb):
        if (m,n) in cc_wake.columns:
            subplot(gs[i+1,j+1])
            plot(cc_wake[(m,n)])
            xticks([0])
            yticks([])
show()






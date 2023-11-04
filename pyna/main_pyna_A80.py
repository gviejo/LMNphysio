# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-12-16 14:24:56
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-11-03 19:26:52
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


path = '/mnt/ceph/users/gviejo/OPTO/A8000/A8054/A8054-230718A'
#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220614A'

data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('rate', 0.6)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

opto_ep = loadOptoEp(path, epoch=1, n_channels = 2, channel = 0)

opto_ep = opto_ep.merge_close_intervals(0.03)

stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

peth = nap.compute_perievent(spikes, nap.Ts(opto_ep["start"].values), minmax=(-stim_duration, 2*stim_duration))

frates = pd.DataFrame({n:np.sum(peth[n].count(0.05), 1).values for n in peth.keys()})

rasters = {j:pd.concat([peth[j][i].as_series().fillna(i) for i in peth[j].index]) for j in peth.keys()}

tcurves = tuning_curves
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

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
        # if SI.loc[i,'SI'] > 0.1:
        #     fill_between(tuning_curves[i].index.values, np.zeros_like(tuning_curves[i].values), tuning_curves[i].values)
        # legend()

        count+=1
        gca().set_xticklabels([])

hd = SI[SI>0.5].dropna().index.values
nhd = SI[SI<0.5].dropna().index.values

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
        title(os.path.basename(path)+"_"+str(n))
        yticks([])
        subplot(subgs[1,1])
        plot(rasters[n], '.', markersize = 0.24)        
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


sys.exit()


opto_ep = loadOptoEp(path, epoch=1, n_channels = 2, channel = 0)

wake2_ep = wake_ep.loc[[0]].set_diff(opto_ep.merge_close_intervals(10.0))

# wake2_ep = nap.IntervalSet(
#     start = data.position.time_support.loc[0,'start'],
#     end = opto_ep.loc[0,'start']
#     )


tc_opto = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = opto_ep)
tc_opto = smoothAngularTuningCurves(tc_opto, window = 20, deviation = 2.0)


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = wake2_ep)
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, wake2_ep, minmax=(0,2*np.pi))

figure()
count = 1
for l,j in enumerate(np.unique(shank)):
    neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
    for k,i in enumerate(neurons):      
        subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection='polar')
        #subplot(1,2,i+1,projection='polar')  
        plot(tuning_curves[i], label = str(shank[l]) + ' ' + str(i), color = colors[l])
        plot(tc_opto[i], '--', label = 'opto', color = colors[l])
        count+=1
        gca().set_xticklabels([])
        legend()
# show()


stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

peth = nap.compute_perievent(spikes, nap.Ts(opto_ep["start"].values), minmax=(-stim_duration, 2*stim_duration))

frates = pd.DataFrame({n:np.sum(peth[n].count(0.05), 1).values for n in peth.keys()})

rasters = {j:pd.concat([peth[j][i].as_series().fillna(i) for i in peth[j].index]) for j in peth.keys()}




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
        plot(tc_opto[i], '--', label = 'opto', color = colors[l])
        legend()
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




def computeAngularVelocity(angle, ep, bin_size):
    """this function only works for single epoch
    """        
    tmp = np.unwrap(angle.restrict(ep).values)
    tmp = pd.Series(index=angle.restrict(ep).index.values, data=tmp)
    tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)    
    tmp = nap.Tsd(t = tmp.index.values, d = tmp.values)    
    tmp = tmp.bin_average(bin_size)
    t = tmp.index.values[0:-1]+np.diff(tmp.index.values)
    velocity = nap.Tsd(t=t, d = np.diff(tmp))
    return velocity

def computeLMN_TC(spikes, angle, ep, velocity):
    atitc = {}    
    bins_velocity   = np.array([velocity.min(), -np.pi/12, np.pi/12, velocity.max()+0.001])
    for n in spikes.index:
        vel = velocity.restrict(ep)
        spkvel = spikes[n].restrict(ep).value_from(velocity)
        idx = np.digitize(spkvel.values, bins_velocity)-1
        tcvel = []
        for k in range(3):
            spk = nap.TsGroup({0:nap.Tsd(spkvel.as_series().iloc[idx == k], time_support = ep)})
            tc = nap.compute_1d_tuning_curves(spk, angle, 60, minmax=(0, 2*np.pi), ep = ep)            
            tc = smoothAngularTuningCurves(tc, 50, 3)
            tcvel.append(tc)
        tcvel = pd.concat(tcvel, 1)
        atitc[n] = tcvel
    return atitc

velocity = computeAngularVelocity(angle, wake_ep, 0.1)
atiopto = computeLMN_TC(spikes, angle, opto_ep, velocity)
atitc = computeLMN_TC(spikes, angle, wake2_ep, velocity)

# figure()
# gs = GridSpec(3, 2)
# for i,n in enumerate(atitc.keys()):    
#     for k in range(3):
#         subplot(gs[k,i])
#         plot(atitc[n].iloc[:,k])
#         plot(atiopto[n].iloc[:,k], '--')
# show()

name = ['clockwise', 'immobile', 'counterclockwise']

figure(figsize = (12, 3))
for k in range(3):        
    subplot(1,3,k+1)
    plot(atitc[0].iloc[:,k])
    plot(atiopto[0].iloc[:,k], '--', label = 'opto')
    title(name[k])
    xlabel("HD")
    ylabel("FR")
    legend()
    

show()
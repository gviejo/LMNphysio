# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-12-16 14:24:56
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-14 14:16:03
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


path = '/mnt/Data2/Opto/A8000/A8047/A8047-230311A'
#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220614A'

data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('rate', 0.6)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']


opto_ep = loadOptoEp(path, epoch=1, n_channels = 2, channel = 0)

wake2_ep = wake_ep.loc[[0]]


tc_opto = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = opto_ep)
# tc_opto = smoothAngularTuningCurves(tc_opto, window = 20, deviation = 2.0)


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = wake2_ep)
# tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)


SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)


stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)
peth = nap.compute_perievent(spikes, nap.Ts(opto_ep["start"].values), minmax=(-4, 14))
frates = pd.DataFrame({n:peth[n].count(1.0).sum(1) for n in peth.keys()})
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
        plot(tc_opto[i], '--', label = 'opto', color = colors[l])
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
        plot(tc_opto[n], '--', label = 'opto', color = colors[l])
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

for i in range(len(hd)):
    figure(figsize = (12, 3))
    for k in range(3): 
        subplot(1,3,k+1)
        plot(atitc[hd[i]].iloc[:,k])
        plot(atiopto[hd[i]].iloc[:,k], '--', label = 'opto')
        title(name[k])
        xlabel("HD")
        ylabel("FR")
        legend()
    

show()


###################
# CORRELATION
sys.exit()

tokeep = hd

wak_rate = zscore_rate(spikes[tokeep].count(0.3, wake_ep))
nopto_rate = zscore_rate(spikes[tokeep].count(0.03, sleep_ep.loc[[0]]))
opto_rate = zscore_rate(spikes[tokeep].count(0.03, opto_ep))

r_wak = np.corrcoef(wak_rate.values.T)[np.triu_indices(len(tokeep),1)]
r_nopto = np.corrcoef(nopto_rate.values.T)[np.triu_indices(len(tokeep),1)]
r_opto = np.corrcoef(opto_rate.values.T)[np.triu_indices(len(tokeep),1)]

r = pd.DataFrame(data = np.vstack((r_wak, r_nopto, r_opto)).T)

pairs = list(combinations(tokeep, 2))

r.index = pd.Index(pairs)
r.columns = pd.Index(['wak', 'nopto', 'opto'])



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue']
shank = shank.flatten()


figure()
subplot(121)
plot(r['wak'], r['opto'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(r['wak'].values, r['opto'].values, 1)
x = np.linspace(r['wak'].min(), r['wak'].max(),5)
plot(x, x*m + b, label = 'opto r='+str(np.round(m,3)), color = 'red')
xlabel('wake')
plot(r['wak'], r['nopto'], 'o', color = 'grey', alpha = 0.5)
m, b = np.polyfit(r['wak'].values, r['nopto'].values, 1)
x = np.linspace(r['wak'].min(), r['wak'].max(),5)
plot(x, x*m + b, label = 'non-opto r='+str(np.round(m,3)), color = 'grey')
legend()
subplot(122)

[plot([0,1],r.loc[p,['nopto', 'opto']], 'o-') for p in r.index]
xticks([0,1], ['sws', 'opto'])

legend()


figure()
ax = subplot(111)
for s, e in opto_ep.values:
    axvspan(s, e, color = 'green', alpha=0.1)

for i,n in enumerate(peaks[hd].sort_values().index.values):
    plot(spikes[n].restrict(sws_ep).fillna(i), '|', 
        markersize = 20, markeredgewidth=2)

show()

datatosave = {
    'frates':frates,
    'r':r
}
import _pickle as cPickle
cPickle.dump(datatosave, open(
    os.path.join('/home/guillaume/Dropbox/CosyneData', 'OPTO_SLEEP_A8047.pickle'), 'wb'
    ))


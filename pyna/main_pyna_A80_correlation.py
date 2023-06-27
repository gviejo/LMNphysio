# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-12-16 14:24:56
# @Last Modified by:   gviejo
# @Last Modified time: 2023-04-12 16:37:09
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


path = '/media/guillaume/New Volume/A8000/A8049/A8049-230412A'
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

opto_ep = loadOptoEp(path, epoch=1, n_channels = 2, channel = 0)

opto_ep = opto_ep.merge_close_intervals(0.03)

tcurves = tuning_curves
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))


stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)
peth = nap.compute_perievent(spikes, nap.Ts(opto_ep["start"].values), minmax=(-stim_duration, 2*stim_duration))
frates = pd.DataFrame({n:peth[n].count(0.01).sum(1) for n in peth.keys()})
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
            fill_between(tuning_curves[i].index.values, np.zeros_like(tuning_curves[i].values), tuning_curves[i].values, color=colors[l])
        legend()

        count+=1
        gca().set_xticklabels([])

hd = SI[SI>0.1].dropna().index.values
nhd = SI[SI<0.1].dropna().index.values

###################
# CORRELATION

tokeep = hd

opto_ep = opto_ep.intersect(sws_ep)

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


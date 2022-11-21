# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-08-10 17:16:25
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-09-02 17:05:47
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
from functions import *
import sys
from itertools import combinations, product
from matplotlib.pyplot import *



#path = '/mnt/Data2/LMN-PSB-2/A3019/A3019-220701A'
#path = '/mnt/Data2/LMN-PSB-2/A3019/A3019-220630A'

#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220613A'
#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220614A'
#path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220614B'
# path = '/mnt/Data2/LMN-PSB-2/A3018/A3018-220615A'
#path = '/mnt/Data2/LMN-PSB-2/A3019/A3019-220629A'
#path = '/mnt/Data2/LMN-PSB-2/A3019/A3019-220630A'
path = '/mnt/Data2/LMN-PSB-2/A3019/A3019-220701A'


data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('freq', 1.0)
angle = data.position['ry']
position = data.position

# turning by pi
tmp = np.unwrap(position['ry'].values)
tmp += np.pi
tmp = np.mod(tmp, 2*np.pi)
angle = nap.Tsd(t = position.index.values, d = tmp, time_support=position.time_support)

wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')
rem_ep = data.read_neuroscope_intervals('rem')
up_ep = data.read_neuroscope_intervals('up')
down_ep = data.read_neuroscope_intervals('down')    


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
spikes.set_info(halfr = r)

spikes2 = spikes.getby_threshold('SI', 0.15).getby_threshold('halfr', 0.25)

tokeep = spikes2.keys()


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
        if i in tokeep:
            fill_between(tuning_curves[i].index.values,
                np.zeros_like(tuning_curves[i].values),
                tuning_curves[i].values, 
                color = colors[l])

show()


#############################
# DECODING
#############################

#spikes = spikes[tokeep]

tcurves = tuning_curves#[tokeep]

wake_ep = wake_ep.loc[[0]]

angle_wak, proba_angle_wak = nap.decode_1d(tuning_curves[tokeep], spikes2, wake_ep, 0.3, feature = angle.restrict(wake_ep.loc[[0]]))

angle_sws, proba_sws = nap.decode_1d(tuning_curves[tokeep], spikes2, sws_ep, 0.04)

angle_rem, _ = nap.decode_1d(tuning_curves[tokeep], spikes2, rem_ep, 0.3)


#############################
# RASTER
#############################

peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

groups = spikes._metadata.groupby("location").groups
psb = groups['psb']
lmn = groups['lmn']

lmn = peaks[lmn].sort_values().index.values
psb = peaks[psb].sort_values().index.values


###########################################################################
#SAVING
###########################################################################
datatosave = { 'wak':angle_wak,
              'rem':angle_rem,
              'sws':angle_sws,
              'tcurves':tcurves,
              'angle':angle,
              'peaks':peaks,
              'spikes':spikes,
              'up_ep':up_ep,
              'down_ep':down_ep,
              'tokeep':tokeep
          }

import _pickle as cPickle
cPickle.dump(datatosave, open('../figures/figures_adrien_2022/fig_1_decoding.pickle', 'wb'))



# wake
figure()
ax = subplot(211)
for i,n in enumerate(psb):
    plot(spikes[n].restrict(wake_ep).fillna(peaks[n]), '|', markersize = 10)
    plot(angle)
    plot(angle_wak, '--')
subplot(212, sharex = ax)
for i,n in enumerate(lmn):
    plot(spikes[n].restrict(wake_ep).fillna(peaks[n]), '|', markersize = 10)
    plot(angle)
    plot(angle_wak, '--')



# rem
figure()
#ex_rem = nts.IntervalSet(start = 1.57085e+10, end = 1.57449e+10)
ax = subplot(211)
for i,n in enumerate(psb):
    if n not in tokeep:      
      plot(spikes[n].restrict(rem_ep).fillna(peaks[n]), '|', color = 'grey', markersize = 10, alpha=0.3) 
    else:
      plot(spikes[n].restrict(rem_ep).fillna(peaks[n]), '|', markersize = 15, markeredgewidth=4)

tmp2 = smoothAngle(angle_rem, 1)
plot(tmp2, '--', linewidth = 2, color = 'black')

subplot(212, sharex = ax)
for i,n in enumerate(lmn):
      plot(spikes[n].restrict(rem_ep).fillna(peaks[n]), '|', markersize = 15, markeredgewidth=4)

plot(tmp2, '--', linewidth = 2, color = 'black')
    


# sws

sws2_ep = sws_ep.loc[[(sws_ep["end"] - sws_ep["start"]).sort_values().index[-1]]]

figure()
ax = subplot(211)
for i,n in enumerate(psb):
    if n not in tokeep:      
      plot(spikes[n].restrict(sws2_ep).fillna(peaks[n]), '|', color = 'grey', markersize = 10) 
    else:
      plot(spikes[n].restrict(sws2_ep).fillna(peaks[n]), '|', markersize = 15, markeredgewidth=4)
tmp2 = smoothAngle(angle_sws, 1)
#plot(tmp2, '--', linewidth = 2, color = 'darkgrey')
plot(tmp2.restrict(up_ep.intersect(sws2_ep)), '.--', color = 'grey')
for s, e in down_ep.intersect(sws2_ep).values:
    axvspan(s, e, color = 'blue', alpha=0.1)
ylim(0, 2*np.pi)

subplot(212, sharex = ax)
for i,n in enumerate(lmn):
    plot(spikes[n].restrict(sws2_ep).fillna(peaks[n]), '|', markersize = 10, markeredgewidth=3)
#plot(tmp2, '--', linewidth = 2, color = 'darkgrey')
plot(tmp2.restrict(up_ep.intersect(sws2_ep)), '.--', color = 'grey')
for s, e in down_ep.intersect(sws2_ep).values:
    axvspan(s, e, color = 'blue', alpha=0.1)
ylim(0, 2*np.pi)    
    
# subplot(313, sharex=ax)
# imshow(proba_sws.restrict(sws2_ep).values.T[::-1],
#       extent=(sws2_ep.start[0], sws2_ep.end[0], 0, 2*np.pi),
#       aspect='auto')

show()



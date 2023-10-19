# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-08-10 17:16:25
# @Last Modified by:   gviejo
# @Last Modified time: 2023-10-18 13:46:57
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
sys.path.append("../")
from functions import *
import sys
from itertools import combinations, product
from matplotlib.pyplot import *


if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'

# path = '/mnt/DataRAID2/LMN-ADN/A5043/A5043-230301A'
path = os.path.join(data_directory, 'LMN-ADN/A5043/A5043-230306A')


data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('rate', 1)
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


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
spikes.set_info(halfr = r)


adn = spikes.getby_category("location")['adn'].getby_threshold('SI', 0.2).getby_threshold('halfr', 0.5).index
lmn = spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.1).getby_threshold('halfr', 0.5).index

tokeep = list(adn) + list(lmn)

spikes2 = spikes[tokeep]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = spikes._metadata.group.values

figure()
count = 1
for l,j in enumerate(np.unique(shank)):
    neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
    for k,i in enumerate(neurons):      
        subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')    
        plot(tuning_curves[i], label = str(np.round(SI.loc[i].values[0], 4)), color = colors[l])
        title(i)
        legend()
        count+=1
        gca().set_xticklabels([])
        if i in tokeep:
            fill_between(tuning_curves[i].index.values,
                np.zeros_like(tuning_curves[i].values),
                tuning_curves[i].values, 
                color = colors[l])

show()

# sys.exit()

#############################
# DECODING
#############################

#spikes = spikes[tokeep]

tcurves = tuning_curves

wake_ep = wake_ep.loc[[0]]

angle_wak, proba_angle_wak = nap.decode_1d(tuning_curves[lmn], spikes2[lmn], wake_ep, 0.3, feature = angle.restrict(wake_ep.loc[[0]]))

angle_sws, proba_sws = nap.decode_1d(tuning_curves[lmn], spikes2[lmn], sws_ep, 0.02)

angle_rem, proba_rem = nap.decode_1d(tuning_curves[lmn], spikes2[lmn], rem_ep, 0.3)


#############################
# RASTER
#############################

peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

adn = peaks[adn].sort_values().index.values
lmn = peaks[lmn].sort_values().index.values

# sys.exit()

###########################################################################
#SAVING
# ###########################################################################
datatosave = { 'wak':angle_wak,
              'rem':angle_rem,
              'sws':angle_sws,
              'tcurves':tcurves,
              'angle':angle,
              'peaks':peaks,
              'spikes':spikes,
              'adn':adn,
              'lmn':lmn,
              'p_rem':proba_rem,
              'p_sws':proba_sws,
              'p_wak':proba_angle_wak,
              # 'up_ep':up_ep,
              # 'down_ep':down_ep,
              'tokeep':tokeep,
              'ex_sws':nap.IntervalSet(
                start = 1522.73,
                end = 1530.38
                ),
              'ex_rem':nap.IntervalSet(
                start = 7600.00,
                end = 7627.00
                ),
              'ex_wak':nap.IntervalSet(
                start = 4560.50,
                end = 4600.00
                )
          }

import _pickle as cPickle
# cPickle.dump(datatosave, open('../figures/figures_adrien_2022/fig_1_decoding.pickle', 'wb'))
# filepath = os.path.join(os.path.expanduser("~"), 'Dropbox/LMNphysio/data/DATA_FIG_2_LMN_ADN_A5043_MS5.pickle')

filepath = os.path.join(os.path.expanduser("~"), 'Dropbox/LMNphysio/data/DATA_FIG_LMN_ADN_{}.pickle'.format(os.path.basename(path)))

cPickle.dump(datatosave, open(filepath, 'wb'))

sws2_ep = sws_ep.loc[[(sws_ep["end"] - sws_ep["start"]).sort_values().index[-2]]]

# wake
figure()
ax = subplot(313)
plot(angle)
title("wwake")
# plot(angle_wak, '--')
subplot(311, sharex = ax)
for i,n in enumerate(adn):
    plot(spikes[n].restrict(wake_ep).fillna(i), '|', markersize = 10)
subplot(312, sharex = ax)
for i,n in enumerate(lmn):
    plot(spikes[n].restrict(wake_ep).fillna(i), '|', markersize = 10)

# sws
figure()
ax = subplot(313)
plot(angle_sws.restrict(sws_ep), '--')
title("sws")
subplot(311, sharex = ax)
for i,n in enumerate(adn):
    plot(spikes[n].restrict(sws_ep).fillna(i), '|', markersize = 15, markeredgewidth=5)
subplot(312, sharex = ax)
for i,n in enumerate(lmn):
    plot(spikes[n].restrict(sws_ep).fillna(i), '|', markersize = 15, markeredgewidth=5)
show()

# rem
figure()
ax = subplot(311)
plot(angle_rem, '--')
title("rem")
subplot(312, sharex = ax)
for i,n in enumerate(adn):
    plot(spikes[n].restrict(rem_ep).fillna(i), '|', markersize = 10)
subplot(313, sharex = ax)
for i,n in enumerate(lmn):
    plot(spikes[n].restrict(rem_ep).fillna(i), '|', markersize = 10)

show()


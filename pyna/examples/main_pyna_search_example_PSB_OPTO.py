# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-11-01 13:15:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-23 18:03:05
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
path = os.path.join(data_directory, 'OPTO/A8000/A8054/A8054-230718A')

basename = os.path.basename(path)
filepath = os.path.join(path, "kilosort4", basename + ".nwb")

nwb = nap.load_file(filepath)

spikes = nwb['units']
spikes = spikes.getby_threshold("rate", 1)

position = []
columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
for k in columns:
    position.append(nwb[k].values)
position = np.array(position)
position = np.transpose(position)
position = nap.TsdFrame(
    t=nwb['x'].t,
    d=position,
    columns=columns,
    time_support=nwb['position_time_support'])

epochs = nwb['epochs']
wake_ep = epochs[epochs.tags == "wake"]
sws_ep = nwb['sws']
rem_ep = nwb['rem']
opto_ep = nwb['opto']


angle = position['ry']

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
spikes.set_info(halfr = r)
spikes.set_info(peak = tuning_curves.idxmax())
spikes.set_info(order = tuning_curves.idxmax().sort_values().index.values)

hd = spikes[spikes.location=='psb'].getby_threshold('SI', 0.2).getby_threshold('halfr', 0.5).index
nhd = np.sort(np.array(list(set(spikes.keys()) - set(hd))))

tokeep = spikes.keys()




colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = spikes.metadata.group.values

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
        if i in hd:
            fill_between(tuning_curves[i].index.values,
                np.zeros_like(tuning_curves[i].values),
                tuning_curves[i].values, 
                color = colors[l])




# figure()
# ax = subplot(211)
# plot(spikes[hd].to_tsd().restrict(sws_ep), '|', markersize=10, mew=3)
# [axvspan(s, e, alpha=0.2) for s, e in opto_ep.values]
# ax = subplot(212, sharex=ax)
# plot(spikes[nhd].to_tsd().restrict(sws_ep), '|', markersize=10, mew=3)
# [axvspan(s, e, alpha=0.2) for s, e in opto_ep.values]



# plot(position['ry'].restrict(wake_ep.loc[[1]]))


stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

pehd = nap.compute_perievent(spikes[hd], opto_ep.starts, (-stim_duration, stim_duration*2))

frates = nap.compute_eventcorrelogram(spikes, nap.Ts(opto_ep.start), stim_duration/20., stim_duration*2, norm=True)

modu = (frates.loc[0.0:1.0].mean(0) - frates.loc[-1.0:0.0].mean(0))/(frates.loc[0.0:1.0].mean(0) + frates.loc[-1.0:0.0].mean(0))


figure()
for i, n in enumerate(hd):
	subplot(int(np.sqrt(len(hd)))+1,int(np.sqrt(len(hd)))+1, i+1)
	plot(pehd[n].to_tsd(), '.')
	title(n)

penhd = nap.compute_perievent(spikes[nhd], opto_ep.starts, (-stim_duration, stim_duration*2))

figure()
for i, n in enumerate(nhd):
	subplot(int(np.sqrt(len(nhd)))+1,int(np.sqrt(len(nhd)))+1, i+1)
	plot(penhd[n].to_tsd(), '.')
	title(n)	

show()



datatosave = {
	'hd':hd,
	'nhd':nhd,
	'tc':tuning_curves,
    'mod':modu,
    'peth':nap.compute_perievent(spikes, opto_ep.starts, (-stim_duration, stim_duration*2))
	}

import _pickle as cPickle

filepath = os.path.join(os.path.expanduser("~"), 'Dropbox/LMNphysio/data/DATA_FIG_PSB_SLEEP_{}.pickle'.format(os.path.basename(path)))

cPickle.dump(datatosave, open(filepath, 'wb'))

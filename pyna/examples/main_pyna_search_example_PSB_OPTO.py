# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-11-01 13:15:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-11-01 17:02:25
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


data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('rate', 1)
angle = data.position['ry']
position = data.position

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
spikes.set_info(pfd = tuning_curves.idxmax())
spikes.set_info(order = tuning_curves.idxmax().sort_values().index.values)

hd = spikes.getby_category("location")['PSB'].getby_threshold('SI', 0.1).getby_threshold('halfr', 0.5).index
nhd = np.sort(np.array(list(set(spikes.keys()) - set(hd))))

tokeep = spikes.keys()

opto_ep = nap.load_file(os.path.join(path, os.path.basename(path))+"_opto_ep.npz")


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
        if i in hd:
            fill_between(tuning_curves[i].index.values,
                np.zeros_like(tuning_curves[i].values),
                tuning_curves[i].values, 
                color = colors[l])




# figure()
# # subplot(211)
# plot(spikes[hd].to_tsd("pfd").restrict(wake_ep.loc[[1]]), '|', markersize=10, mew=3)
# [axvspan(s, e, alpha=0.2) for s, e in opto_ep.values]
# plot(position['ry'].restrict(wake_ep.loc[[1]]))


stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

pehd = nap.compute_perievent(spikes[hd], opto_ep.starts, (-stim_duration, stim_duration*2))

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
sys.exit()

datatosave = {
	'hd':pehd[16],
	'nhd':penhd[15],
	'tc':tuning_curves[[16,15]]
	}

import _pickle as cPickle

filepath = os.path.join(os.path.expanduser("~"), 'Dropbox/LMNphysio/data/DATA_FIG_PSB_SLEEP_{}.pickle'.format(os.path.basename(path)))

cPickle.dump(datatosave, open(filepath, 'wb'))

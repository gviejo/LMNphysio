# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-02-24 16:49:32
import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *

############################################################################################### 
# GENERAL infos
###############################################################################################
if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)


for s in datasets:
# for s in ['LMN-PSB/A3018/A3018-220613A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

    if not os.path.exists(filepath):
        print(f"\nMissing ks4 nwb for {s}\n")
    else:    
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
        nwb.close()

        if "adn" in spikes.location.values:
            spikes = spikes[spikes.location=="adn"]

        if "psb" in spikes.location.values:
            spikes = spikes[spikes.location=="psb"]
        

        #################################################################################################
        #DETECTION UP/DOWN States
        #################################################################################################
        total = spikes.count(0.01, sws_ep).sum(1)/0.01
        total2 = total.smooth(2*0.01)
        
        down_ep = total2.threshold(np.percentile(total2, 20), method='below').time_support
        down_ep = down_ep.merge_close_intervals(0.25)
        down_ep = down_ep.drop_short_intervals(0.05)
        down_ep = down_ep.drop_long_intervals(2)

        up_ep = sws_ep.set_diff(down_ep)
        top_ep = total2.threshold(np.percentile(total2, 80), method='above').time_support

        write_neuroscope_intervals(path, basename, 'up', up_ep)
        write_neuroscope_intervals(path, basename, 'down', down_ep)
        write_neuroscope_intervals(path, basename, 'top', top_ep)



figure()
ax = subplot(211)
for n in spikes.index:
    plot(spikes[n].restrict(sws_ep).fillna(n), '|')
subplot(212, sharex =ax)
plot(total2.restrict(sws_ep))
plot(total2.restrict(down_ep), '.', color = 'blue')
plot(total2.restrict(top_ep), '.', color = 'red')
show()


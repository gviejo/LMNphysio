# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-30 18:46:13
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
# data_directory = '/mnt/DataGuillaume/'

# infos = getAllInfos(data_directory, datasets)

data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt('/mnt/DataRAID2/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
# datasets = ['LMN-PSB-2/A3019/A3019-220701A']
infos = getAllInfos(data_directory, datasets)

# sys.exit()

for s in datasets:
#for s in ['LMN-PSB/A3019/A3019-220701A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes.getby_threshold('rate', 0.4)
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    angle = position['ry']
    tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    spikes.set_info(SI)
    r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
    spikes.set_info(halfr = r)

    psb = spikes.getby_category("location")['psb'].getby_threshold('SI', 0.4).getby_threshold('halfr', 0.5).index
    lmn = spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.2).getby_threshold('halfr', 0.5).index

    tokeep = psb

    spikes = spikes[tokeep]

    
    #################################################################################################
    #DETECTION STAGE 1/ STAGE 2 States
    #################################################################################################
    total = spikes.count(0.5, sws_ep).sum(1)/0.5
    total2 = total.rolling(window=40,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
    total2 = nap.Tsd(total2, time_support = sws_ep)
    sta2_ep = total2.threshold(np.percentile(total2, 25), method='below').time_support
    sta2_ep = sta2_ep.drop_short_intervals(1)
    # sta2_ep = sta2_ep.drop_long_intervals(2)
    sta1_ep = sws_ep.set_diff(sta2_ep)
    
    data.write_neuroscope_intervals('.sta1.evt', sta1_ep, 'PySta1')
    data.write_neuroscope_intervals('.sta2.evt', sta2_ep, 'PySta2')    



figure()
ax = subplot(211)
for n in spikes.index:
    plot(spikes[n].restrict(sws_ep).fillna(n), '|')
subplot(212, sharex =ax)
plot(total2.restrict(sws_ep))
plot(total2.restrict(sta2_ep), '.', color = 'blue')
show()


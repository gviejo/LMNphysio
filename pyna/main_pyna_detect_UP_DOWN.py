# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-07-07 14:18:50
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

data_directory = '/mnt/Data2/'
datasets = np.genfromtxt('/mnt/DataGuillaume/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
# datasets = ['LMN-PSB-2/A3019/A3019-220701A']
infos = getAllInfos(data_directory, datasets)


for s in datasets:

    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')

    idx = spikes._metadata[spikes._metadata["location"].str.contains("psb")].index.values
    spikes = spikes[idx]

    #################################################################################################
    #DETECTION UP/DOWN States
    #################################################################################################
    total = spikes.count(0.01, sws_ep).sum(1)/0.01
    total2 = total.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
    total2 = nap.Tsd(total2, time_support = sws_ep)
    down_ep = total2.threshold(np.percentile(total2, 20), method='below').time_support
    up_ep = sws_ep.set_diff(down_ep)

    data.write_neuroscope_intervals('.up.evt', up_ep, 'PyUp')
    data.write_neuroscope_intervals('.down.evt', down_ep, 'PyDown')

    
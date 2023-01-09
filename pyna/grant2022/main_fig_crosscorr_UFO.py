# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-08-09 15:41:55
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-08-09 16:51:06

import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
sys.path.append("../")
from functions import *
import pynacollada as pyna
# from ufo_detection import *

data_dir = ['/mnt/Data2/', '/mnt/DataGuillaume', '/mnt/DataGuillaume']

dataset_list = ['datasets_LMN_PSB.list', 'datasets_LMN_ADN.list', 'datasets_LMN.list']

allcc = {s:[] for s in ['lmn', 'adn', 'psb']}

for i in range(len(dataset_list)):

    ############################################################################################### 
    # GENERAL infos
    ###############################################################################################
    data_directory = '/mnt/DataGuillaume/'
    datasets = np.genfromtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')
    

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
        #rem_ep = data.read_neuroscope_intervals('rem')
        try:
            ufo_ep, ufo_ts = loadUFOs(path)
        except:
            ufo_ep, ufo_ts = None,None

        if len(ufo_ts):
            groups = data.spikes._metadata.groupby("location").groups

            grp = np.intersect1d(['adn', 'lmn', 'psb'], list(groups.keys()))

            for st in grp:
                idx = data.spikes._metadata[data.spikes._metadata["location"].str.contains(st)].index.values
                spikes = data.spikes[idx]

                ############################################################################################### 
                # COMPUTING EVENT CORRELOGRAM
                ###############################################################################################
                ufo_cc = nap.compute_eventcorrelogram(spikes, ufo_ts, 0.05, 2, sws_ep)

                allcc[st].append(ufo_cc)

sys.exit()

cc = pd.concat(cc, 1)

figure()
plot(cc, alpha =0.4, color = 'grey')
plot(cc.mean(1), linewidth=3)
show()
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-11-29 14:59:45
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-15 17:07:00
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.stats import zscore
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt('/mnt/DataRAID2/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)



allr = []

# for s in datasets:
for s in ['LMN-PSB/A3019/A3019-220701A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes.getby_threshold('rate', 0.75)
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    up_ep = read_neuroscope_intervals(data.path, data.basename, 'up')
    down_ep = read_neuroscope_intervals(data.path, data.basename, 'down')


    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.002).time_support         

    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|psb")].index.values
    spikes = spikes[idx]
        
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tcurves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tcurves = smoothAngularTuningCurves(tcurves, 20, 4)
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    spikes.set_info(SI, peaks=peaks)

    psb = list(spikes.getby_category("location")["psb"].getby_threshold("SI", 0.06).index)
    lmn = list(spikes.getby_category("location")["lmn"].getby_threshold("SI", 0.1).index)
    nhd = list(spikes.getby_category("location")["psb"].getby_threshold("SI", 0.04, op='<').index)



    # figure()
    # for i, n in enumerate(spikes.index):
    #     subplot(10,10,i+1, projection='polar')        
    #     if n in psb:
    #         plot(tcurves[n], color = 'red')
    #     elif n in lmn:
    #         plot(tcurves[n], color = 'green')
    #     elif n in nhd:
    #         plot(tcurves[n], color = 'blue')
    #     else:
    #         plot(tcurves[n], color = 'grey')
    #     xticks([])
    #     yticks([])

    # sys.exit()

    


    ############################################################################################### 
    # PEER PREDICTION
    ###############################################################################################
    zwake = {}
    zsws = {}
    for gr, grp in zip(['psb', 'lmn', 'nhd'], [psb, lmn, nhd]):
        count = spikes[grp].count(0.3, wake_ep)
        zref = StandardScaler().fit_transform(count)
        zwake[gr] = zref

        count = spikes[grp].count(0.03, sws_ep)
        ztar = StandardScaler().fit_transform(count)
        zsws[gr] = zref

    
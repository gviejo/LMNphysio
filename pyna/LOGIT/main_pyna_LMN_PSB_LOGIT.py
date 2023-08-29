# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-23 16:07:32
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt('/mnt/DataRAID2/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)



eta = {'lmn':[], 'psb':[]}

for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')

    rem_ep = read_neuroscope_intervals(data.path, data.basename, 'rem')
    up_ep = read_neuroscope_intervals(data.path, data.basename, 'up')
    down_ep = read_neuroscope_intervals(data.path, data.basename, 'down')
    top_ep = read_neuroscope_intervals(data.path, data.basename, 'top')
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|psb")].index.values
    spikes = spikes[idx]
        
    # spikes = spikes.getby_category("location")['lmn']


      
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
    
    # CHECKING HALF EPOCHS
    wake2_ep = splitWake(position.time_support.loc[[0]])    
    tokeep2 = []
    stats2 = []
    tcurves2 = []   
    for i in range(2):
        tcurves_half = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

        tokeep, stat = findHDCells(tcurves_half)
        tokeep2.append(tokeep)
        stats2.append(stat)
        tcurves2.append(tcurves_half)       
    tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
    
    spikes = spikes[tokeep]    
    tcurves         = tuning_curves[tokeep]
    peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.002).time_support     

    lmn = list(spikes._metadata.groupby("location").groups['lmn'])
    psb = list(spikes._metadata.groupby("location").groups['psb'])


    ############################################################################################### 
    # LOGIT
    ###############################################################################################
    if len(lmn) > 3:
        p = {}        
        bin_size = 0.3
        up_gr = nap.TsGroup({0:nap.Ts(up_ep["start"].values)}, time_support = sws_ep)

        for g, gr in zip(['lmn', 'psb'], [lmn, psb]):
            rate_wak = zscore_rate(spikes[gr].count(bin_size, newwake_ep)/bin_size)
            rate_shu = zscore_rate(nap.randomize.shuffle_ts_intervals(spikes[gr]).count(bin_size, newwake_ep)/bin_size)
            rate_sws = zscore_rate(spikes[gr].count(0.02, sws_ep)/0.02)

            for rate in [rate_wak, rate_shu, rate_sws]:
                rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)


            X = np.vstack((rate_wak.values, rate_shu.values))
            y = np.hstack((np.zeros(len(rate_wak)), np.ones(len(rate_shu))))
            Xt = rate_sws.values

            # clf = LogisticRegression(random_state=0).fit(X, y)
            bst = XGBClassifier(n_estimators=10, max_depth=10, learning_rate=0.001, objective='binary:logistic')
            bst.fit(X, y)

            tmp = 1.0 - bst.predict_proba(Xt)[:,0]

            p = nap.Tsd(t = rate_sws.index.values, d = tmp, time_support = sws_ep)

            tmp = nap.compute_event_trigger_average(up_gr, p, 0.03, (-1,2), sws_ep)
            eta[g].append(tmp[0])


for g in eta.keys():
    eta[g] = pd.concat(eta[g], 1)
    eta[g] = eta[g].apply(zscore)

figure()
plot(eta['lmn'].mean(1), label = 'lmn')
plot(eta['psb'].mean(1), label = 'psb')
legend()
show()
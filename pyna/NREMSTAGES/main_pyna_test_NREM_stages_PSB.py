# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-16 11:33:44
import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *
import pynacollada as pyna
from scipy.signal import filtfilt
from scipy.stats import zscore



############################################################################################### 
# GENERAL infos
###############################################################################################
# data_directory = '/mnt/DataGuillaume/'
data_directory = '/mnt/DataRAID2/'
# data_directory = '/media/guillaume/LaCie'


datasets = np.genfromtxt('/mnt/DataRAID2/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
# On Razer
# datasets = np.genfromtxt('/media/guillaume/LaCie/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')

durations = {2:[], 3:[]}

datatosave = {}

allr = []
pearson = {}


for s in datasets:
# for s in ['LMN-PSB/A3019/A3019-220701A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    # data.spikes = data.spikes.getby_threshold('rate', 0.4)
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    angle = position['ry']
    tuning_curves = nap.compute_1d_tuning_curves(data.spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    data.spikes.set_info(SI)
    r = correlate_TC_half_epochs(data.spikes, angle, 120, (0, 2*np.pi))
    data.spikes.set_info(halfr = r)
    tcurves = tuning_curves
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    powers = {}

    groups = {}

    for k, thr in zip(['lmn', 'psb'], [0.1, 0.5]):
        spikes = data.spikes.getby_category("location")[k].getby_threshold('SI', thr).getby_threshold('halfr', 0.5)
    
        tokeep = spikes.index
        groups[k] = tokeep

        ##############################
        # REACTIVATION
        bin_size_wake = 0.3
        bin_size_sws = 0.02

        #  WAKE 
        count = spikes.count(bin_size_wake, wake_ep)
        rate = count/bin_size_wake
        #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        rate = zscore_rate(rate)
        C = (1/rate.shape[0])*np.dot(rate.values.T, rate.values)
        C[np.diag_indices_from(C)] = 0.0

        # SWS
        count = spikes.count(bin_size_sws, sws_ep)
        rate = count/bin_size_sws
        #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        rate = zscore_rate(rate)
        
        p = np.sum(np.dot(rate.values, C) * rate.values, 1)
        power = nap.Tsd(t=count.index.values, d = p, time_support = sws_ep)

        power = power.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=20)
        power = power - power.mean()
        power = power / power.std()
        power = nap.Tsd(t=power.index.values, 
            d = power.values,
            time_support = sws_ep)

        powers[k] = power
        # nrem2_ep = power.threshold(np.percentile(power, 50)).time_support    
        # nrem3_ep = power.threshold(np.percentile(power, 50), 'below').time_support


        # ############################################################################################### 
        # # PEARSON CORRELATION
        # ###############################################################################################
        # rates = {}
        # for e, ep, bin_size, std in zip(
        #         ['wak', 'nrem2', 'nrem3'], 
        #         [wake_ep, nrem2_ep, nrem3_ep], 
        #         [0.2, 0.03, 0.03], 
        #         [1, 1, 1]
        #         ):
        #     count = pd.concat([
        #         spikes[n].count(bin_size, ep) for n in spikes.index
        #         ], 1)
        #     rate = count/bin_size
        #     # rate = rate.as_dataframe()
        #     #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
        #     rate = rate.apply(zscore)            
        #     rates[e] = rate 
        
        # # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
        # pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))
        # pairs = pd.MultiIndex.from_tuples(pairs)
        # r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

        # for ep in rates.keys():
        #     tmp = np.corrcoef(rates[ep].values.T)
        #     r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

        # name = data.basename    
        # pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
        # pairs = pd.MultiIndex.from_tuples(pairs)
        # r.index = pairs
        
        # #######################
        # # COMPUTING PEARSON R FOR EACH SESSION
        # #######################
        # pearson[s] = np.zeros((3))
        # pearson[s][0] = scipy.stats.pearsonr(r['wak'], r['nrem2'])[0]
        # pearson[s][1] = scipy.stats.pearsonr(r['wak'], r['nrem3'])[0]
        # pearson[s][2] = len(spikes)

        # allr.append(r)




figure()

sws2_ep = sws_ep.loc[[(sws_ep["end"] - sws_ep["start"]).sort_values().index[-1]]]

# sws2_ep = nap.IntervalSet(
#     start = 5652, end = 5665
#     )

ax = subplot(311)

axvline(5652)

plot(powers['lmn'].restrict(sws2_ep), label = 'lmn')
plot(powers['psb'].restrict(sws2_ep), label = 'psb')
legend()

subplot(312, sharex = ax)
for i,n in enumerate(peaks[groups['psb']].sort_values().index.values):
    plot(data.spikes[n].restrict(sws2_ep).fillna(i), '|', 
        markersize = 10, markeredgewidth=1)

subplot(313, sharex = ax)
for i,n in enumerate(peaks[groups['lmn']].sort_values().index.values):
    plot(data.spikes[n].restrict(sws2_ep).fillna(i), '|', 
        markersize = 10, markeredgewidth=1)

show()

from scipy.linalg import hankel

def offset_matrix(rate, binsize=0.01, windowsize = 0.1):
    idx1 = -np.arange(0, windowsize + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, windowsize + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))

    # Build the Hankel matrix
    tmp = rate
    n_p = len(idx1)
    n_f = len(idx2)
    pad_tmp = np.pad(tmp, (n_p, n_f))
    offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]        

    return offset_tmp, time_idx





plmn, tindex = offset_matrix(powers['lmn'], 0.02, 1.0)

a = []
for i in range(len(tindex)):
    a.append(scipy.stats.pearsonr(plmn[:,i], powers['psb'].values)[0])
    

plot(tindex, a)
show()
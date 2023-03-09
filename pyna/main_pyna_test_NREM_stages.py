# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   gviejo
# @Last Modified time: 2023-03-08 19:30:47
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
# data_directory = '/mnt/DataRAID2/'
data_directory = '/media/guillaume/LaCie'


# datasets = np.genfromtxt('/mnt/DataRAID2/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
# On Razer
datasets = np.genfromtxt('/media/guillaume/LaCie/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')

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
    tcurves = tuning_curves
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    psb = spikes.getby_category("location")['psb']
    spikes = spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.2).getby_threshold('halfr', 0.5)
    
    lmn = spikes.index
    #################################################################################################
    #DETECTION STAGE 1/ STAGE 2 States
    #################################################################################################

    # # MUA    
    # binsize = 0.1
    # total = spikes.count(binsize, sws_ep)
    # total = total.bin_average(binsize).sum(1)
    # total2 = total.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=10)
    # total2 = nap.Tsd(total2, time_support = sws_ep)
    # total2 = total2 - total2.mean()
    # total2 = total2 / total2.std()
    # power = total2    
    # nrem2_ep = total2.threshold(0).time_support

    # nrem3_ep = total2.threshold(0, method='below').time_support
    # # nrem2_ep = nrem2_ep.merge_close_intervals(binsize*2)

    # # sta2_ep = sta2_ep.drop_long_intervals(2)
    # # nrem3_ep = sws_ep.set_diff(nrem2_ep)

    # nrem2_ep = nrem2_ep.drop_short_intervals(0.1)
    # nrem3_ep = nrem3_ep.drop_short_intervals(0.1)

    # # nrem2_ep = nrem2_ep.drop_long_intervals(10)
    # # nrem3_ep = nrem3_ep.drop_long_intervals(5)


    # # [durations[2].append(v) for v in nrem2_ep['end'] - nrem2_ep['start']]
    # # [durations[3].append(v) for v in nrem3_ep['end'] - nrem3_ep['start']]


    #############################
    # # LFP
    frequency = 1250.0
    binsize = 0.05
    lfp = data.load_lfp(channel=16,extension='.eeg',frequency=1250)
    lfp = downsample(lfp, 1, 5)
    lfp = lfp.restrict(sws_ep)

    signal = pyna.eeg_processing.bandpass_filter(lfp, 0.5, 4.0, 250, order=3)

    power = signal.pow(2).bin_average(binsize)
    power = power.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=10)
    power = power - power.mean()
    power = power / power.std()
    power = nap.Tsd(t=power.index.values, 
        d = power.values,
        time_support = sws_ep)

    nrem3_ep = power.threshold(np.percentile(power, 80)).time_support
    
    nrem2_ep = power.threshold(np.percentile(power, 20), 'below').time_support
    ###############################



    ############################################################################################### 
    # PEARSON CORRELATION
    ###############################################################################################
    rates = {}
    for e, ep, bin_size, std in zip(
            ['wak', 'nrem2', 'nrem3'], 
            [wake_ep, nrem2_ep, nrem3_ep], 
            [0.2, 0.03, 0.03], 
            [1, 1, 1]
            ):
        count = pd.concat([
            spikes[n].count(bin_size, ep) for n in spikes.index
            ], 1)
        rate = count/bin_size
        # rate = rate.as_dataframe()
        #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
        rate = rate.apply(zscore)            
        rates[e] = rate 
    
    # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
    pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))
    pairs = pd.MultiIndex.from_tuples(pairs)
    r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

    for ep in rates.keys():
        tmp = np.corrcoef(rates[ep].values.T)
        r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

    name = data.basename    
    pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
    pairs = pd.MultiIndex.from_tuples(pairs)
    r.index = pairs
    
    #######################
    # COMPUTING PEARSON R FOR EACH SESSION
    #######################
    pearson[s] = np.zeros((3))
    pearson[s][0] = scipy.stats.pearsonr(r['wak'], r['nrem2'])[0]
    pearson[s][1] = scipy.stats.pearsonr(r['wak'], r['nrem3'])[0]
    pearson[s][2] = len(spikes)

    allr.append(r)

allr = pd.concat(allr, 0)

pearson = pd.DataFrame(pearson).T
pearson.columns = ['nrem2', 'nrem3', 'count']



figure()
subplot(131)
plot(allr['wak'], allr['nrem2'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['nrem2'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('nrem2')
r, p = scipy.stats.pearsonr(allr['wak'], allr['nrem2'])
title('r = '+str(np.round(r, 3)))

subplot(132)
plot(allr['wak'], allr['nrem3'], 'o',  alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['nrem3'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('nrem3')
r, p = scipy.stats.pearsonr(allr['wak'], allr['nrem3'])
title('r = '+str(np.round(r, 3)))




figure()

sws2_ep = sws_ep.loc[[(sws_ep["end"] - sws_ep["start"]).sort_values().index[-1]]]

# sws2_ep = nap.IntervalSet(
#     start = 5652, end = 5665
#     )

ax = subplot(311)

for s, e in nrem2_ep.intersect(sws2_ep).values:
    axvspan(s, e, color = 'green', alpha=0.1)
for s, e in nrem3_ep.intersect(sws2_ep).values:
    axvspan(s, e, color = 'orange', alpha=0.1)  

axvline(5652)

plot(power.restrict(sws2_ep))

subplot(312, sharex = ax)
plot(lfp.restrict(sws2_ep))
plot(signal.restrict(sws2_ep))

subplot(313, sharex = ax)
for i,n in enumerate(peaks[lmn].sort_values().index.values):
    plot(spikes[n].restrict(sws2_ep).fillna(i), '|', 
        markersize = 10, markeredgewidth=1)

show()



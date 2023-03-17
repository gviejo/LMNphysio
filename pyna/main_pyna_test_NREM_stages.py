# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-16 12:05:56
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

rates_psb = []
mua_psb = {}

sims = {}


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

    psb = spikes.getby_category("location")['psb'].getby_threshold('SI', 0.7)
    spikes = spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.2).getby_threshold('halfr', 0.5)
    
    lmn = spikes.index

    #################################################################################################
    #DETECTION STAGE 1/ STAGE 2 States
    #################################################################################################

    # # MUA    
    binsize = 0.02
    total = psb.count(binsize, sws_ep)
    total = total.sum(1)
    total2 = total.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
    total2 = nap.Tsd(total2, time_support = sws_ep)
    total2 = total2 - total2.mean()
    total2 = total2 / total2.std()

    total2 = total2.bin_average(0.1)

    ##############################
    # REACTIVATION
    ##############################
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
    
    power = power.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=10)
    power = power - power.mean()
    power = power / power.std()
    power = nap.Tsd(t=power.index.values, 
        d = power.values,
        time_support = sws_ep)

    power = power.bin_average(0.1)

    nrem2_ep = power.threshold(np.percentile(power, 50)).time_support    
    nrem3_ep = power.threshold(np.percentile(power, 50), 'below').time_support

    nrem2_ep = nrem2_ep.drop_short_intervals(0.1)
    nrem3_ep = nrem3_ep.drop_short_intervals(0.1)

    rates_psb.append(
        pd.DataFrame(
            {
                'nrem2':psb.restrict(nrem2_ep).rates,
                'nrem3':psb.restrict(nrem3_ep).rates
            }))

    mua_psb[data.basename] = {
        'nrem2':psb.count(0.05, nrem2_ep).sum().sum() / nrem2_ep.tot_length("s"),
        'nrem3':psb.count(0.05, nrem3_ep).sum().sum() / nrem3_ep.tot_length("s")
    }

    


    ############################################################################################### 
    # PEARSON CORRELATION
    ###############################################################################################
    rates = {}
    for e, ep, bin_size, std in zip(
            ['wak', 'nrem2', 'nrem3'], 
            [wake_ep, nrem2_ep, nrem3_ep], 
            [0.2, 0.02, 0.02], 
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


    #########################
    # COSINE SIMILARITY
    #########################
    
    cossim = {}
    for k in ['wak', 'nrem2', 'nrem3']:
        r = rates[k].values
        nrm = np.linalg.norm(rates[k].values, axis=1)
        coss = []
        for i in range(0,len(rates[k])-1):
            if np.sum(r[i])>-1 and np.sum(r[i+1])>-1:
                coss.append(
                    np.sum(r[i]*r[i+1])/(nrm[i]*nrm[i+1])
                    )
        cossim[k] = np.array(coss)

    sims[s.split("/")[-1]] = cossim

    


allr = pd.concat(allr, 0)

pearson = pd.DataFrame(pearson).T
pearson.columns = ['nrem2', 'nrem3', 'count']

rates_psb = pd.concat(rates_psb)

mua_psb = pd.DataFrame(mua_psb).T

figure()
subplot(131)
# [plot([0, 1], v, 'o-') for v in rates_psb.values]
hist(rates_psb["nrem2"] / rates_psb["nrem3"], 30)
xlim(0, 2)
subplot(132)
hist(mua_psb["nrem2"] / mua_psb["nrem3"])
xlim(0, 2)
subplot(133)
xx = np.linspace(-2, 2, 30)
imshow(np.histogram2d(total2.values, power.values, (xx, xx))[0], origin="lower")


figure()
for j, s in enumerate(sims.keys()):
    subplot(3,3,j+1)
    cossim = sims[s]
    tmp = []
    for i, k in enumerate(cossim.keys()):    
        x, b = np.histogram(cossim[k], np.linspace(-1, 1, 40))
        x = x / x.sum()
        plot(b[0:-1], x, label = k)
    legend()
    title(s)


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

subplot(133)
y = pearson
for j, e in enumerate(['nrem2', 'nrem3']):
    plot(np.ones(len(y))*j + np.random.randn(len(y))*0.1, y[e], 'o', markersize=5)
    plot([j-0.2, j+0.2], [y[e].mean(), y[e].mean()], '-', linewidth=0.75)
xticks([0, 1])
xlim(-0.4,1.4)
ylim(-1, 1)
# print(scipy.stats.ttest_ind(y["rem"], y["sws"]))
print(scipy.stats.wilcoxon(y["nrem2"], y["nrem3"]))




figure()

sws2_ep = sws_ep.loc[[(sws_ep["end"] - sws_ep["start"]).sort_values().index[-1]]]

# sws2_ep = nap.IntervalSet(
#     start = 5652, end = 5665
#     )

ax = subplot(211)

for s, e in nrem2_ep.intersect(sws2_ep).values:
    axvspan(s, e, color = 'green', alpha=0.1)
for s, e in nrem3_ep.intersect(sws2_ep).values:
    axvspan(s, e, color = 'orange', alpha=0.1)  

axvline(5652)

plot(power.restrict(sws2_ep))

subplot(212, sharex = ax)
for i,n in enumerate(peaks[lmn].sort_values().index.values):
    plot(spikes[n].restrict(sws2_ep).fillna(i), '|', 
        markersize = 10, markeredgewidth=1)

show()



# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-08-10 15:12:44
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



############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/Data2/'
datasets = np.genfromtxt('/mnt/DataGuillaume/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)



allr = []

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
    rem_ep = data.read_neuroscope_intervals('rem')
    up_ep = data.read_neuroscope_intervals('up')
    down_ep = data.read_neuroscope_intervals('down')    
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|psb")].index.values
    spikes = spikes[idx]
    
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
    groups = spikes._metadata.loc[tokeep].groupby("location").groups
    
    tcurves         = tuning_curves[tokeep]
    peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.005).time_support 
    
    ############################################################################################### 
    # PEARSON CORRELATION
    ###############################################################################################
    rates = {}
    for e, ep, bin_size, std in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], [0.1, 0.1, 0.02], [2, 2, 2]):
        count = spikes.count(bin_size, ep)
        rate = count/bin_size        
        #rate = zscore_rate(rate)
        rate = rate.apply(zscore)
        rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
        rates[e] = nap.TsdFrame(rate, time_support = ep)

    rates['up'] = rates['sws'].restrict(up_ep)
    rates['down'] = rates['sws'].restrict(down_ep)

    idx=np.sort(np.random.choice(len(rates["sws"]), len(rates["down"]), replace=False))    
    rates['rnd'] = rates['sws'].iloc[idx,:]

    
    # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
    pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))    
    pairs = pd.MultiIndex.from_tuples(pairs)
    r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)
    for p in r.index:
        for ep in rates.keys():
            r.loc[p, ep] = scipy.stats.pearsonr(rates[ep][int(p[0])],rates[ep][int(p[1])])[0]

    name = data.basename
    pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
    pairs = pd.MultiIndex.from_tuples(pairs)
    r.index = pairs
    

    #######################
    # SAVING
    #######################
    allr.append(r)

allr = pd.concat(allr, 0)


datatosave = {'allr':allr}
cPickle.dump(datatosave, open(os.path.join('../data/', 'All_correlation_ADN_LMN.pickle'), 'wb'))

clrs = ['lightgray', 'gray']
names = ['ADN', 'LMN']

mkrs = 6



figure()
subplot(231)
plot(allr['wak'], allr['rem'], 'o',  alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['rem'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('rem')
r, p = scipy.stats.pearsonr(allr['wak'], allr['rem'])
title('r = '+str(np.round(r, 3)))

subplot(232)
plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('sws')
r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
title('r = '+str(np.round(r, 3)))


subplot(233)
plot(allr['wak'], allr['up'], 'o', color='orange', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['up'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('up')

r, p = scipy.stats.pearsonr(allr['wak'], allr['up'])
title('r = '+str(np.round(r, 3)))

subplot(234)
plot(allr['wak'], allr['down'], 'o', color='green', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['down'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('down')
r, p = scipy.stats.pearsonr(allr['wak'], allr['down'])
title('r = '+str(np.round(r, 3)))

subplot(235)
plot(allr['wak'], allr['rnd'], 'o', color='green', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['rnd'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('rnd sws')
r, p = scipy.stats.pearsonr(allr['wak'], allr['rnd'])
title('r = '+str(np.round(r, 3)))


show()

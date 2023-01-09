# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-21 09:12:20
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
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt('/mnt/DataRAID2/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)



allr = []
allf = []

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
    
    # idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|psb")].index.values
    # spikes = spikes[idx]
        
    spikes = spikes.getby_category("location")['lmn']


      
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
    newwake_ep = velocity.threshold(0.002).time_support     

    ############################################################################################### 
    # PEARSON CORRELATION
    ###############################################################################################
    rates = {}
    for e, ep, bin_size, std in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], [0.1, 0.1, 0.03], [1,1,1]):
        count = spikes.count(bin_size, ep)
        rate = count/bin_size        
        #rate = zscore_rate(rate)
        rate = rate.as_dataframe().apply(zscore)
        rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
        rates[e] = nap.TsdFrame(rate, time_support = ep)

    rates['up'] = rates['sws'].restrict(up_ep)
    rates['down'] = rates['sws'].restrict(down_ep)
    rates['top'] = rates['sws'].restrict(top_ep)

    idx=np.sort(np.random.choice(len(rates["sws"]), len(rates["down"]), replace=False))    
    rates['rnd'] = rates['sws'].iloc[idx,:]
    
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

    ############################################################################################### 
    # RATES
    ###############################################################################################
    tmp = {}    
    for e, ep in zip(['wak', 'rem', 'sws', 'down', 'up', 'top'], [wake_ep, rem_ep, sws_ep, down_ep, up_ep, top_ep]):
        tmp[e] = spikes.restrict(ep).rates
    allf.append(pd.DataFrame.from_dict(tmp))
    #######################
    # SAVING
    #######################
    allr.append(r)

allr = pd.concat(allr, 0)
allf = pd.concat(allf, 0)

datatosave = {'allr':allr}
cPickle.dump(datatosave, open(os.path.join('../data/', 'All_correlation_ADN_LMN.pickle'), 'wb'))

clrs = ['lightgray', 'gray']
names = ['ADN', 'LMN']

mkrs = 6

allr = allr.dropna()

rval = {}
axes = []

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
rval['wake vs rem'] = r
axes.append(gca())

subplot(232)
plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('sws')
r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
title('r = '+str(np.round(r, 3)))
rval['wake vs sws'] = r
axes.append(gca())

subplot(233)
plot(allr['wak'], allr['up'], 'o', color='orange', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['up'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('up')
r, p = scipy.stats.pearsonr(allr['wak'], allr['up'])
title('r = '+str(np.round(r, 3)))
rval['wake vs UP'] = r
axes.append(gca())

subplot(234)
plot(allr['wak'], allr['down'], 'o', color='green', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['down'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('down')
r, p = scipy.stats.pearsonr(allr['wak'], allr['down'])
title('r = '+str(np.round(r, 3)))
rval['wake vs DOWN'] = r
axes.append(gca())

subplot(235)
plot(allr['wak'], allr['top'], 'o', color='green', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['top'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('top')
r, p = scipy.stats.pearsonr(allr['wak'], allr['top'])
title('r = '+str(np.round(r, 3)))
rval['wake vs TOP'] = r
axes.append(gca())

subplot(236)
plot(allr['wak'], allr['rnd'], 'o', color='green', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['rnd'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('rnd sws')
r, p = scipy.stats.pearsonr(allr['wak'], allr['rnd'])
title('r = '+str(np.round(r, 3)))
xlim(-1,1)
ylim(-1,1)
rval['wake vs RANDOM'] = r
axes.append(gca())


xlims = []
ylims = []
for ax in axes:
    xlims.append(ax.get_xlim())
    ylims.append(ax.get_ylim())
xlims = np.array(xlims)
ylims = np.array(ylims)
xl = (np.min(xlims[:,0]), np.max(xlims[:,1]))
yl = (np.min(ylims[:,0]), np.max(ylims[:,1]))
for ax in axes:
    ax.set_xlim(xl)
    ax.set_ylim(xl)



rval = pd.Series(rval).sort_values()
figure()
bar(np.arange(len(rval)), rval.values)
xticks(np.arange(len(rval)), rval.index.values, rotation=20)


figure()
hist(allf["down"] - allf["sws"], alpha=0.4, label = "Down - sws")
hist(allf["top"] - allf["sws"], alpha=0.4, label = "Top - sws")
hist(allf["up"] - allf["sws"], alpha=0.4, label = "Up - sws")
legend()

show()

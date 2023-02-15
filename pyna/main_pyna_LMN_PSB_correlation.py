# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-10 23:18:38
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



allr = {'psb':[], 'lmn':[]}
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
    sws3_ep = read_neuroscope_intervals(data.path, data.basename, 'nrem3')    
    sws2_ep = read_neuroscope_intervals(data.path, data.basename, 'nrem2')

    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|psb")].index.values
    spikes = spikes[idx]
        
    
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    angle = position['ry']
    tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    spikes.set_info(SI)
    r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
    spikes.set_info(halfr = r)

    psb = list(spikes.getby_category("location")['psb'].getby_threshold('SI', 1).getby_threshold('halfr', 0.5).index)
    lmn = list(spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.1).getby_threshold('halfr', 0.5).index)
        
    ############################################################################################### 
    # PEARSON CORRELATION
    ###############################################################################################
    rates = {}
    for e, ep, bin_size, std in zip(
            ['wak', 'rem', 'sws', 'sws2', 'sws3'], 
            [wake_ep, rem_ep, sws_ep, sws2_ep, sws3_ep], 
            [0.1, 0.1, 0.05, 0.05, 0.05],
            [1,1,1,1,1]):
        count = spikes.count(bin_size, ep)
        rate = count/bin_size        
        #rate = zscore_rate(rate)
        rate = rate.as_dataframe().apply(zscore)
        rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
        rates[e] = nap.TsdFrame(rate, time_support = ep)


    for k, neurons in zip(['psb', 'lmn'], [psb, lmn]):
        # pairs = list(product(psb, lmn))
        pairs = list(combinations(neurons, 2))
        pairs = pd.MultiIndex.from_tuples(pairs)
        r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

        for ep in rates.keys():            
            tmp = np.corrcoef(rates[ep][neurons].values.T)
            r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

        name = data.basename
        pairs = list(combinations([name+'_'+str(n) for n in neurons], 2)) 
        pairs = pd.MultiIndex.from_tuples(pairs)
        r.index = pairs
        
        allr[k].append(r)

for k in allr.keys():
    allr[k] = pd.concat(allr[k], 0)



# datatosave = {'allr':allr}
# cPickle.dump(datatosave, open(os.path.join('../data/', 'All_correlation_ADN_LMN.pickle'), 'wb'))

clrs = ['lightgray', 'gray']
names = ['PSB', 'LMN']

mkrs = 6

# allr = allr.dropna()

rval = {'psb':{}, 'lmn':{}}
axes = []

figure()
gs = GridSpec(2, 4)
for i, k in enumerate(['psb', 'lmn']):
    allr[k] = allr[k].dropna()
    for j,e in enumerate(['rem', 'sws', 'sws2', 'sws3']):
        subplot(gs[i,j])
        plot(allr[k]['wak'], allr[k][e], 'o',  alpha = 0.5)
        m, b = np.polyfit(allr[k]['wak'].values, allr[k][e].values, 1)
        x = np.linspace(allr[k]['wak'].min(), allr[k]['wak'].max(), 4)
        plot(x, x*m + b)
        xlabel('wake')
        ylabel(e)
        r, p = scipy.stats.pearsonr(allr[k]['wak'], allr[k][e])
        title('r = '+str(np.round(r, 3)))
        rval[k]['wake vs '+e] = r
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




rval = pd.DataFrame(rval)
figure()
bar(np.arange(len(rval)), rval['psb'].values, 0.2, label = 'psb')
bar(np.arange(len(rval))+0.25, rval['lmn'].values, 0.2, label = 'lmn')
xticks(np.arange(len(rval))+0.125, rval.index.values, rotation=20)
legend()

show()

# figure()
# hist(allf["down"] - allf["sws"], alpha=0.4, label = "Down - sws")
# hist(allf["top"] - allf["sws"], alpha=0.4, label = "Top - sws")
# hist(allf["up"] - allf["sws"], alpha=0.4, label = "Up - sws")
# legend()

# show()

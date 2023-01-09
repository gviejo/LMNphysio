# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-21 11:19:21

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
datasets = []
for lst in ['datasets_LMN.list', 'datasets_LMN_PSB.list', 'datasets_LMN_ADN.list']:
    datasets.append(np.genfromtxt(os.path.join(data_directory,lst), delimiter = '\n', dtype = str, comments = '#'))    
datasets = np.unique(np.hstack(datasets))

allr = []
pearson = {}

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
    
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
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
    
    if len(tokeep) > 3:

        spikes = spikes[tokeep]
        # groups = spikes._metadata.loc[tokeep].groupby("location").groups
        tcurves         = tuning_curves[tokeep]
        peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

        velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.005).time_support 
        newwake_ep = newwake_ep.drop_short_intervals(0.5)
        ############################################################################################### 
        # PEARSON CORRELATION
        ###############################################################################################
        rates = {}
        for e, ep, bin_size, std in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], [0.1, 0.1, 0.01], [2, 2, 2]):
            count = spikes.count(bin_size, ep)
            rate = count/bin_size
            rate = rate.as_dataframe()
            rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)       
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
        pearson[s][0] = scipy.stats.pearsonr(r['wak'], r['rem'])[0]
        pearson[s][1] = scipy.stats.pearsonr(r['wak'], r['sws'])[0]
        pearson[s][2] = len(spikes)

        #######################
        # SAVING
        #######################
        allr.append(r)

allr = pd.concat(allr, 0)

pearson = pd.DataFrame(pearson).T
pearson.columns = ['rem', 'sws', 'count']

datatosave = {
    'allr':allr,
    'pearsonr':pearson
    }
cPickle.dump(datatosave, open(os.path.join('/home/guillaume/Dropbox/CosyneData', 'All_correlation_LMN.pickle'), 'wb'))


figure()
subplot(131)
plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
xlabel('wake')
ylabel('sws')
r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
title('r = '+str(np.round(r, 3)))

subplot(132)
plot(allr['wak'], allr['rem'], 'o',  alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['rem'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
xlabel('wake')
ylabel('rem')
r, p = scipy.stats.pearsonr(allr['wak'], allr['rem'])
title('r = '+str(np.round(r, 3)))

show()

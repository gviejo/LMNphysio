# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-08 15:49:57
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations, product
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from scipy.linalg import hankel
from sklearn.linear_model import PoissonRegressor
from scipy.ndimage import gaussian_filter

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

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

stds = [0, 5]

binsizes = {
    0:          0.002,
    stds[1]:    0.015}
windowsizes = {
    0:          0.05, 
    stds[1]:    2.0}

coefs_mua = {e:{s:[] for s in stds} for e in ['wak', 'sws', 'rem']}
coefs_pai = {e:{s:[] for s in stds} for e in ['wak', 'sws', 'rem']}

pairs_info = pd.DataFrame(columns = ['offset', 'session'])

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
    down_ep = data.read_neuroscope_intervals('down')
    idx = spikes._metadata[spikes._metadata["location"].str.contains("adn|lmn")].index.values
    spikes = spikes[idx]
      
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves)    
    tcurves = tuning_curves
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    spikes.set_info(SI, peaks=peaks)

    adn = list(spikes.getby_category("location")["adn"].getby_threshold("SI", 0.5).index)
    lmn = list(spikes.getby_category("location")["lmn"].getby_threshold("SI", 0.1).index)

    if len(lmn) > 6 and len(adn) > 6:

        tokeep = adn+lmn
        tokeep = np.array(tokeep)
        spikes = spikes[tokeep]    

        velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.001).time_support 

        tcurves         = tuning_curves[tokeep]
        peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

        ############################################################################################### 
        # GLM
        ###############################################################################################

        pairs = list(product(adn, lmn))

        groups = spikes.getby_category("location")

        for s in stds:

            binsize = binsizes[s]
            windowsize = windowsizes[s]

            count = spikes.count(binsize)
            # rate = count.values
            rate = count/binsize
            
            if s:            
                frate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=stds[1])
                rates = StandardScaler().fit_transform(frate.values)
            else:            
                fratehigh = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1.5)
                rates = StandardScaler().fit_transform((rate - fratehigh).values)

            rates = nap.TsdFrame(
                t=count.index.values, d=rates, 
                columns = count.columns.values,
                time_support = count.time_support)
            
            for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):

                for i, p in enumerate(pairs):

                    pair_name = data.basename + '_' + str(p[0]) + '_' + str(p[1])                
                    a = peaks[p[1]] - peaks[p[0]]
                    pair_offset = np.abs(np.arctan2(np.sin(a), np.cos(a)))
                    pairs_info.loc[pair_name, 'offset'] = pair_offset
                    pairs_info.loc[pair_name, 'session'] = s

                    rate = rates[list(p)].restrict(ep).values
                    
                    reg_offset, time_idx = offset_matrix(rate[:,1], binsize, windowsize)

                    target = np.vstack(rate[:,0])
                    target = np.tile(target, reg_offset.shape[1])
                    
                    r = np.sum(target * reg_offset, 0)

                    # r = r / r.sum()

                    # sys.exit()

                    r = r - r.mean()
                    r = r / r.std()

                    coefs_pai[e][s].append(pd.DataFrame(
                        index=time_idx, data=r, columns=[pair_name]))


pairs_info = pairs_info.sort_values(by="offset")



for k in coefs_mua.keys():
    for s in coefs_mua[k].keys():
        # coefs_mua[k][s] = pd.concat(coefs_mua[k][s], 1)
        coefs_pai[k][s] = pd.concat(coefs_pai[k][s], 1)        
        # coefs_mua[k][s] = coefs_mua[k][s][pairs_info.index]
        coefs_pai[k][s] = coefs_pai[k][s][pairs_info.index]




inters = np.linspace(0, np.pi, 4)
idx = np.digitize(pairs_info['offset'], inters)-1


figure()
gs = GridSpec(4,3)

for i,st in enumerate(stds):

    for j, k in enumerate(['wak', 'rem', 'sws']):
        subplot(gs[0+2*i,j])
        tmp = coefs_pai[k][st]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)        
        tmp = gaussian_filter(tmp.values.T, (2,1))
        imshow(tmp, aspect='auto', cmap = 'jet')
        title(k)

        subplot(gs[1+2*i,j])
        for l in range(3):
            tmp = coefs_pai[k][st].iloc[:,idx==l]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            plot(tmp.mean(1), 'o-', markersize=2)
            # plot(tmp, '-')



betaratio = {}
for i,e in enumerate(coefs_pai.keys()): # epoch 
    betaratio[e] = {}
    for k, s in enumerate(coefs_pai[e].keys()): # Std
        betaratio[e][s] = {}
        betaneg = coefs_pai[e][s].loc[-0.2:0].abs().sum(0)
        betapos = coefs_pai[e][s].loc[0:0.2].abs().sum(0)
        betar = betapos/betaneg
        for j, g in enumerate(np.unique(idx)):
            betaratio[e][s][g] = betar[idx==g]
            

figure()
colors = ['blue', 'orange', 'green']
gs = GridSpec(2,1)
for k, s in enumerate(coefs_pai[e].keys()): # Std
    subplot(gs[k,0])
    count = 0
    axhline(1)
    for i,e in enumerate(coefs_pai.keys()): # epoch         
        for j, g in enumerate(np.unique(idx)): # group
            y = betaratio[e][s][g].values
            x = np.ones(len(y))*count + np.random.randn(len(y))*0.1
            plot(x, y, 'o', color = colors[j], alpha=0.2)
            plot([count-0.2, count + 0.2], [y.mean(), y.mean()], color = colors[l], linewidth=1)
            count += 1
        count += 2
    xticks([1, 5, 11], ['wak', 'rem', 'sws'])


show()
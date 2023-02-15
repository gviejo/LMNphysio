# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-03 17:43:10
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

stds = [0, 2, 5]

coefs = {e:{s:[] for s in stds} for e in ['wak', 'sws', 'rem']}
# coefs_shu = {'wak':[], 'sws':[], 'rem':[]}

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

    # figure()
    # for i, n in enumerate(spikes.index):
    #     subplot(10,10,i+1, projection='polar')        
    #     if n in adn:
    #         plot(tcurves[n], color = 'red')
    #     elif n in lmn:
    #         plot(tcurves[n], color = 'green')
    #     else:
    #         plot(tcurves[n], color = 'grey')
    #     xticks([])
    #     yticks([])

    # sys.exit()
    if len(lmn) > 3 and len(adn) > 3:

        tokeep = adn+lmn
        tokeep = np.array(tokeep)
        spikes = spikes[tokeep]    

        velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.001).time_support 

        ############################################################################################### 
        # GLM
        ###############################################################################################
        groups = spikes.getby_category("location")

        ## MUA ########
        mua = {
            0:nap.Ts(t=np.sort(np.hstack([groups['adn'][j].index.values for j in groups['adn'].index]))),
            1:nap.Ts(t=np.sort(np.hstack([groups['lmn'][j].index.values for j in groups['lmn'].index])))}

        mua = nap.TsGroup(mua, time_support = spikes.time_support, location = np.array(['adn', 'lmn']))

        for e, ep, binsize, windowsize in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], 
                                                    [0.1, 0.1, 0.01],
                                                    [2.0, 2.0, 0.5]):
            for st in stds:
                count = mua.count(binsize, ep)                
                rate = count/binsize
                if st:
                    rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=st)
                rate = StandardScaler().fit_transform(rate.values)

                offset_tmp, time_idx = offset_matrix(rate[:,1], binsize, windowsize)

                target = count.values[:,0]

                glm = PoissonRegressor(max_iter = 1000)
                glm.fit(offset_tmp, target)
                coefs[e][st].append(pd.DataFrame(
                    index=time_idx, data=glm.coef_[0:len(time_idx)], columns=[s])
                    )

                # ################### 
                # # Randomization
                # ###################
                # count = nap.randomize.shuffle_ts_intervals(mua).count(binsize, ep)
                # rate = count.values
                # # rate = count/binsize
                # # rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
                # # rate = StandardScaler().fit_transform(rate)
                # offset_tmp, time_idx = offset_matrix(rate[:,1], binsize, windowsize)

                # glm = PoissonRegressor(max_iter = 1000)
                # glm.fit(offset_tmp, rate[:,0])
                # coefs_shu[e].append(pd.Series(index=time_idx, data=glm.coef_))        



for k in coefs.keys():
    for s in coefs[k].keys():
        coefs[k][s] = pd.concat(coefs[k][s], 1)    

betaratio = pd.DataFrame(
        index=list(coefs['wak'].keys()),
        columns=list(coefs.keys()))

for i,e in enumerate(coefs.keys()):
    for k, s in enumerate(coefs[e].keys()):    
        betaneg = coefs[e][s].loc[-0.4:0].abs().sum(0)
        betapos = coefs[e][s].loc[0:0.4].abs().sum(0)
        betar = betapos/betaneg
        betaratio.loc[s,e] = betar.mean()
            

figure()
for k in betaratio.columns:
    plot(betaratio[k], label = k)
legend()
show()


figure()
gs = GridSpec(2,3)
for i, k in enumerate(['wak', 'rem', 'sws']):
    subplot(gs[0,i])
    plot(coefs[k])
    plot(coefs[k].mean(1))
    title(k)
    subplot(gs[1,i])
    plot(coefs[k].mean(1))
    plot(coefs_shu[k].mean(1), label = 'shuffled')
    legend()


show()
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-01-23 17:40:32
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-14 15:20:15

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')

coefs_mua = {k:{e:[] for e in ['wake', 'rem', 'sws']} for k in ['psb', 'lmn']}
coefs_pai = {k:{e:[] for e in ['wake', 'rem', 'sws']} for k in ['psb', 'lmn']}

pairs_info = {k:pd.DataFrame(columns = ['offset', 'session']) for k in ['psb', 'lmn']}

alltc = []
allsi = []


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
    sws2_ep = read_neuroscope_intervals(data.path, data.basename, 'nrem2')
    sws3_ep = read_neuroscope_intervals(data.path, data.basename, 'nrem3')

    try:
        maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
        
    except:
        meanwf, maxch = data.load_mean_waveforms(spike_count=100)
        maxch.to_csv(data.nwb_path + "/maxch.csv")        


    angle = position['ry']
    tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    spikes.set_info(SI)
    r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
    spikes.set_info(halfr = r)

    psb = list(
        spikes.getby_category("location")['psb']
        .getby_threshold('rate', 0.6)
        .getby_threshold('SI', 0.8)
        .getby_threshold('halfr', 0.5).index)
    lmn = list(spikes.getby_category("location")['lmn']
        .getby_threshold('SI', 0.1)
        .getby_threshold('halfr', 0.5).index)
      
    if len(lmn) > 4 and len(psb) > 5:

        tokeep = psb+lmn
        tokeep = np.array(tokeep)
        spikes = spikes[tokeep]   

        velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.001).time_support 

        tcurves         = tuning_curves[tokeep]
        peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

        ############################################################################################### 
        # GLM
        ###############################################################################################
        for k, neurons in zip(['psb', 'lmn'], [psb, lmn]):

            # pairs = list(product(psb, lmn))
            pairs = list(combinations(neurons, 2))

            pairs = [p for p in pairs if np.abs(maxch[p[0]] - maxch[p[1]]) > 2]
                        
            for i, p in enumerate(pairs):
                tar_neuron = p[0]
                reg_neuron = p[1]
                
                pair_name = data.basename + '_' + str(p[0]) + '_' + str(p[1])

                a = peaks[tar_neuron] - peaks[reg_neuron]
                pair_offset = np.abs(np.arctan2(np.sin(a), np.cos(a)))        
                pairs_info[k].loc[pair_name, 'offset'] = pair_offset
                pairs_info[k].loc[pair_name, 'session'] = s
             
                ## MUA ########
                group = list(set(tokeep) - set([reg_neuron]))
                mua = {0:nap.Ts(t=np.sort(np.hstack([spikes[j].index.values for j in group])))}
                mua = nap.TsGroup(mua, time_support = spikes.time_support, location = np.array([k]))

                for e, ep, binsize, windowsize in zip(
                        ['wake', 'rem', 'sws'], 
                        [wake_ep, rem_ep, sws_ep], 
                        [0.1, 0.1, 0.02],
                        [2, 2, 0.5]):

                    count = mua[0].count(binsize, ep)
                    # rate = count.values/binsize
                    rate = count/binsize
                    # rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)                    
                    rate = StandardScaler().fit_transform(rate.values[:,np.newaxis])
                    mua_offset, time_idx = offset_matrix(rate.flatten(), binsize, windowsize)

                    count1 = spikes[p[0]].count(binsize, ep)
                    count2 = spikes[p[1]].count(binsize, ep)

                    count = pd.DataFrame(index=count1.index.values, 
                        data = np.vstack((count1.values, count2.values)).T)

                    # rate = count.values/binsize
                    rate = count/binsize
                    # rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
                    rate = StandardScaler().fit_transform(rate.values)
                    neuron_offset, time_idx = offset_matrix(rate[:,1], binsize, windowsize)
                    
                    # # VERSION single GLM
                    # reg_psb = pd.DataFrame(index=time_idx, data=0, columns=[pair_name])
                    # reg_mua = pd.DataFrame(index=time_idx, data=0, columns=[pair_name])

                    # for j, t in enumerate(time_idx):
                    #     glm = PoissonRegressor(max_iter = 100)                    
                    #     glm.fit(
                    #         np.vstack((neuron_offset[:, j], mua_offset[:, j])).T,
                    #         count.values[:,0])
                    #     reg_psb.loc[t] = glm.coef_[0]
                    #     reg_mua.loc[t] = glm.coef_[1]

                    # coefs_pai[k][e].append(reg_psb)
                    # coefs_mua[k][e].append(reg_mua)

                    # Version grouped offset
                    offset_tmp = np.hstack((mua_offset, neuron_offset))
                    glm = PoissonRegressor(max_iter = 1000)                    
                    glm.fit(offset_tmp, count.values[:,0])
                    coefs_mua[k][e].append(pd.DataFrame(
                        index=time_idx, data=glm.coef_[0:len(time_idx)], columns=[pair_name]))
                    coefs_pai[k][e].append(pd.DataFrame(
                        index=time_idx, data=glm.coef_[len(time_idx):], columns=[pair_name]))




for g in ['psb', 'lmn']:
    pairs_info[g] = pairs_info[g].sort_values(by="offset")
    for k in coefs_mua[g].keys():
        coefs_mua[g][k] = pd.concat(coefs_mua[g][k], 1)
        coefs_pai[g][k] = pd.concat(coefs_pai[g][k], 1)        
        coefs_mua[g][k] = coefs_mua[g][k][pairs_info[g].index]
        coefs_pai[g][k] = coefs_pai[g][k][pairs_info[g].index]


# datatosave = {
#     'coefs_mua':coefs_mua,
#     'coefs_pai':coefs_pai,
#     'pairs_info':pairs_info
#     }

# cPickle.dump(datatosave, open(
#     os.path.join('/home/guillaume/Dropbox/CosyneData', 'GLM_BETA_WITHIN.pickle'), 'wb'
#     ))



figure()
gs = GridSpec(4,3)

for i, g in enumerate(['psb', 'lmn']):
    inters = np.linspace(0, np.pi, 4)
    idx = np.digitize(pairs_info[g]['offset'], inters)-1

    for j, k in enumerate(['wake', 'rem', 'sws']):
        subplot(gs[i*2,j])
        tmp = coefs_pai[g][k]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
        tmp = gaussian_filter(tmp.values.T, (2,1))
        imshow(tmp, aspect='auto', cmap = 'jet')
        title(k)
    
        subplot(gs[i*2+1,j])
        for l in range(3):
            tmp = coefs_pai[g][k].iloc[:,idx==l]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            plot(tmp.mean(1), '-')            


figure()
gs = GridSpec(4,3)

for i, g in enumerate(['psb', 'lmn']):
    inters = np.linspace(0, np.pi, 4)
    idx = np.digitize(pairs_info[g]['offset'], inters)-1

    for j, k in enumerate(['wake', 'rem', 'sws']):
        subplot(gs[i*2,j])
        tmp = coefs_mua[g][k]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
        tmp = gaussian_filter(tmp.values.T, (2,1))
        imshow(tmp, aspect='auto', cmap = 'jet')
        title(k)
    
        subplot(gs[i*2+1,j])
        for l in range(3):
            tmp = coefs_mua[g][k].iloc[:,idx==l]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            plot(tmp.mean(1), '-')            



show()        



        



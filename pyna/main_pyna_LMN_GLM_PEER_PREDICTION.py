# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-01-23 17:40:32
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-06 17:00:33

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
from scipy.io import loadmat

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


coefs_mua = {e:[] for e in ['wak', 'sws2', 'sws3']}
coefs_pai = {e:[] for e in ['wak', 'sws2', 'sws3']}

pairs_info = pd.DataFrame(columns = ['offset', 'session'])

SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.5
    }

k = 'lmn'

for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    wake_ep = data.epochs['wake']
    sleep_ep = data.epochs['sleep']
 
    position = data.position

    sws_ep = data.read_neuroscope_intervals('sws')        
    rem_ep = data.read_neuroscope_intervals('rem')

    sws3_ep = read_neuroscope_intervals(data.path, data.basename, 'nrem3')    
    sws2_ep = read_neuroscope_intervals(data.path, data.basename, 'nrem2')

    try:
        maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
        
    except:
        meanwf, maxch = data.load_mean_waveforms(spike_count=1000)
        maxch.to_csv(data.nwb_path + "/maxch.csv")        


    idx = spikes._metadata[spikes._metadata["location"].str.contains('lmn')].index.values
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

    tokeep = list(spikes.getby_category("location")[k].getby_threshold("SI", SI_thr[k]).getby_threshold("rate", 0.5).index)
    

    if len(tokeep) > 5:
        
        spikes = spikes[tokeep]

        # velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
        # newwake_ep = velocity.threshold(0.001).time_support 

        tcurves         = tuning_curves[tokeep]
        peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

        ############################################################################################### 
        # GLM
        ###############################################################################################

        # pairs = list(product(adn, lmn))
        pairs = list(combinations(tokeep, 2))

        if k == 'adn':
            pairs = [p for p in pairs if np.abs(maxch[p[0]] - maxch[p[1]]) > 1]
        if k == 'lmn':
            pairs = [p for p in pairs if np.abs(maxch[p[0]] - maxch[p[1]]) > 2]
        
        groups = spikes.getby_category("location")

        for i, p in enumerate(pairs):
            tar_neuron = p[0]
            reg_neuron = p[1]
            
            pair_name = data.basename + '_' + str(p[0]) + '_' + str(p[1])

            a = peaks[tar_neuron] - peaks[reg_neuron]
            pair_offset = np.abs(np.arctan2(np.sin(a), np.cos(a)))        
            pairs_info.loc[pair_name, 'offset'] = pair_offset
            pairs_info.loc[pair_name, 'session'] = s
         
            ## MUA ########
            group = list(set(tokeep) - set([reg_neuron]))
            mua = {0:nap.Ts(t=np.sort(np.hstack([groups[k][j].index.values for j in group])))}
            mua = nap.TsGroup(mua, time_support = spikes.time_support, location = np.array([k]))

            for e, ep, binsize, windowsize in zip(
                    ['wak', 'sws2', 'sws3'], 
                    [wake_ep, sws2_ep, sws3_ep], 
                    [0.1, 0.01, 0.01],
                    [3.0, 1.0, 1.0]):

                count = mua.count(binsize, ep)
                # rate = count.values/binsize
                rate = count/binsize
                rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
                rate = StandardScaler().fit_transform(rate.values)
                mua_offset, time_idx = offset_matrix(rate.flatten(), binsize, windowsize)

                count = spikes[list(p)].count(binsize, ep)
                # rate = count.values/binsize
                rate = count/binsize
                rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
                rate = StandardScaler().fit_transform(rate.values)
                neuron_offset, time_idx = offset_matrix(rate[:,1], binsize, windowsize)
                
                # VERSION single GLM
                # reg_adn = pd.DataFrame(index=time_idx, data=0, columns=[pairs_offset[p]])
                # reg_mua = pd.DataFrame(index=time_idx, data=0, columns=[pairs_offset[p]])

                # for j, t in enumerate(time_idx):
                #     glm = PoissonRegressor(max_iter = 100)                    
                #     glm.fit(
                #         np.vstack((adn_offset[:, j], mua_offset[:, j])).T,
                #         count.values[:,0])
                #     reg_adn.loc[t] = glm.coef_[0]
                #     reg_mua.loc[t] = glm.coef_[1]

                # coefs_pai[e].append(reg_adn)
                # coefs_mua[e].append(reg_mua)

                # Version grouped offset
                offset_tmp = np.hstack((mua_offset, neuron_offset))
                glm = PoissonRegressor(max_iter = 100)                    
                glm.fit(offset_tmp, count.values[:,0])
                coefs_mua[e].append(pd.DataFrame(
                    index=time_idx, data=glm.coef_[0:len(time_idx)], columns=[pair_name]))
                coefs_pai[e].append(pd.DataFrame(
                    index=time_idx, data=glm.coef_[len(time_idx):], columns=[pair_name]))





pairs_info = pairs_info.sort_values(by="offset")
for k in coefs_mua.keys():
    coefs_mua[k] = pd.concat(coefs_mua[k], 1)
    coefs_pai[k] = pd.concat(coefs_pai[k], 1)        
    coefs_mua[k] = coefs_mua[k][pairs_info.index]
    coefs_pai[k] = coefs_pai[k][pairs_info.index]


# datatosave = {
#     'coefs_mua':coefs_mua,
#     'coefs_pai':coefs_pai,
#     'pairs_info':pairs_info
#     }

# cPickle.dump(datatosave, open(
#     os.path.join('/home/guillaume/Dropbox/CosyneData', 'GLM_BETA_ALL_WITHIN.pickle'), 'wb'
#     ))


figure()
gs = GridSpec(3,3)

inters = np.linspace(0, np.pi, 4)
idx = np.digitize(pairs_info['offset'], inters)-1

for j, k in enumerate(['wak', 'sws2', 'sws3']):
    subplot(gs[0,j])
    tmp = coefs_pai[k]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
    tmp = gaussian_filter(tmp.values.T, (2,1))
    imshow(tmp, aspect='auto', cmap = 'jet')
    title(k)

    subplot(gs[1,j])
    for l in range(3):
        tmp = coefs_pai[k].iloc[:,idx==l]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        plot(tmp.mean(1), '-')            


figure()
gs = GridSpec(6,3)

for i, g in enumerate(structs):
    inters = np.linspace(0, np.pi, 4)
    idx = np.digitize(pairs_info[g]['offset'], inters)-1

    for j, k in enumerate(['wak', 'rem', 'sws']):
        subplot(gs[i*2,j])
        tmp = coefs_mua[g][k]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
        tmp = gaussian_filter(tmp.values.T, (2,1))
        imshow(tmp, aspect='auto', cmap = 'jet')
        title(k)
    
        subplot(gs[i*2+1,j])
        for l in range(3):
            tmp = coefs_mua[g][k].iloc[:,idx==l]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            plot(tmp.mean(1), '-')            


figure()
gs = GridSpec(2, 3)
for i, g in enumerate(structs):
    for j, k in enumerate(['rem', 'sws']):
        subplot(gs[j,i])        
        plot(coefs_pai[g]['wak'].loc[0], coefs_pai[g][k].loc[0], 'o')
        m, b = np.polyfit(coefs_pai[g]['wak'].loc[0].values, coefs_pai[g][k].loc[0].values, 1)
        x = np.linspace(coefs_pai[g]['wak'].loc[0].min(), coefs_pai[g]['wak'].loc[0].max(),5)
        plot(x, x*m + b)    
        xlabel("wake")
        ylabel(k)
        title(g)


show()



        



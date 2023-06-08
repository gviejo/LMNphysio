# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-06-07 20:23:55
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

from GLM_HMM import fit_pop_glm, GLM_HMM

# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
# from sklearn.decomposition import PCA, FastICA, KernelPCA
# from sklearn.manifold import Isomap
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import f1_score


############################################################################################### 
# GENERAL infos
###############################################################################################
# data_directory = '/mnt/DataRAID2/'
data_directory = '/mnt/ceph/users/gviejo'
# data_directory = '/media/guillaume/LaCie'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')


SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.5
    }

allr = []

# for s in datasets:
for s in ['LMN-ADN/A5002/A5002-200304A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    if os.path.isdir(os.path.join(path, "pynapplenwb")):

        data = nap.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake'].loc[[0]]
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')
        # down_ep = data.read_neuroscope_intervals('down')
        idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
        spikes = spikes[idx]
          
        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves)    
        tcurves = tuning_curves
        SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
        spikes.set_info(SI)
        spikes.set_info(max_fr = tcurves.max())

        spikes = spikes.getby_threshold("SI", SI_thr["lmn"])
        spikes = spikes.getby_threshold("rate", 1.0)
        spikes = spikes.getby_threshold("max_fr", 3.0)

        tokeep = spikes.index
        tcurves = tcurves[tokeep]
        peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        order = peaks.sort_values().index.values
        spikes.set_info(order=order, peaks=peaks)


        if len(tokeep) > 7:
            
            # figure()
            # for i in range(len(tokeep)):
            #     subplot(4, 4, i+1, projection='polar')
            #     plot(tcurves[tokeep[i]])
            # show()
            
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1).merge_close_intervals(1)

            ############################################################################################### 
            # HMM GLM
            ###############################################################################################
            
            bin_size = 0.01
            nbasis = 8
            nt = 200

            Ws = []
            W2s = []

            # GLM 0            

            W, W2 = fit_pop_glm(spikes, newwake_ep, bin_size*1, nt, nbasis)
            Ws.append(W)
            W2s.append(W2)

            # GLM 1

            spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(wake_ep))
            W, W2 = fit_pop_glm(spikes2, newwake_ep, bin_size*1, nt, nbasis)
            Ws.append(W)
            W2s.append(W2)


            # figure()
            # gs = GridSpec(2, nbasis)
            # for i in range(2):
            #     for j in range(nbasis):
            #         subplot(gs[i, j])
            #         tmp = pd.DataFrame(data=W2s[i][:,j,:], index = spikes.keys(), columns=spikes.keys())
            #         imshow(tmp.loc[order,order], aspect='auto')
            # show()

            hmm = GLM_HMM(Ws, nt, nbasis)            

            hmm.fit(spikes, sws_ep, bin_size)

            time_idx = spikes[spikes.index[0]].count(bin_size, sws_ep).index.values

            figure()
            ax = subplot(311)        
            plot(time_idx, hmm.bestZ)
            # plot(position['ry'].restrict(newwake_ep))

            subplot(312, sharex=ax)
            plot(spikes.restrict(sws_ep).to_tsd("peaks"), '|', markersize=10)

            subplot(313, sharex=ax)
            plot(time_idx, hmm.O)
            # show()

            Z = nap.Tsd(t=time_idx, d=hmm.bestZ, time_support=sws_ep)

            ep1 = Z.threshold(0.5).time_support.drop_short_intervals(0.05)
            ep0 = Z.threshold(0.5, 'below').time_support.drop_short_intervals(0.05)


            ############################################################################################### 
            # PEARSON CORRELATION
            ###############################################################################################        
            rates = {}
            for e, ep, bin_size, std in zip(['wake', 'sws'], [newwake_ep, sws_ep], [0.3, 0.02], [1, 2]):
                count = spikes.count(bin_size, ep)
                rate = count/bin_size
                rate = rate.as_dataframe()
                rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)       
                rate = rate.apply(zscore)                    
                rates[e] = nap.TsdFrame(rate)
            
            rates['ep0'] = rates['sws'].restrict(ep0)
            rates['ep1'] = rates['sws'].restrict(ep1)
            _ = rates.pop("sws")

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
            # SAVING
            #######################
            allr.append(r)

allr = pd.concat(allr, 0)



figure()
subplot(131)
plot(allr['wake'], allr['ep0'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wake'].values, allr['ep0'].values, 1)
x = np.linspace(allr['wake'].min(), allr['wake'].max(),5)
plot(x, x*m + b)
xlabel('wakee')
ylabel('ep0')
r, p = scipy.stats.pearsonr(allr['wake'], allr['ep0'])
title('r = '+str(np.round(r, 3)))

subplot(132)
plot(allr['wake'], allr['ep1'], 'o',  alpha = 0.5)
m, b = np.polyfit(allr['wake'].values, allr['ep1'].values, 1)
x = np.linspace(allr['wake'].min(), allr['wake'].max(), 4)
plot(x, x*m + b)
xlabel('wakee')
ylabel('ep1')
r, p = scipy.stats.pearsonr(allr['wake'], allr['ep1'])
title('r = '+str(np.round(r, 3)))

show()

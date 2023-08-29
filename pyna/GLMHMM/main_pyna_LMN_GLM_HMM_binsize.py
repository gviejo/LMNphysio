# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-13 14:51:09
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
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PoissonRegressor
from GLM_HMM import GLM_HMM
from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
from sklearn.preprocessing import StandardScaler


############################################################################################### 
# GENERAL infos
###############################################################################################
# data_directory = '/mnt/DataRAID2/'
data_directory = '/mnt/ceph/users/gviejo'
# data_directory = '/media/guillaume/LaCie'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

# datasets = np.hstack([
#     np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
#     np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
#     np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
#     ])


SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.5
    }



bins = [0.02, 0.05, 0.1, 0.2, 0.3]
rb = pd.DataFrame(index=bins, columns=np.arange(2))

for bin_size in bins:

    allr = []

    for s in datasets:
    #for s in ['LMN-ADN/A5002/A5002-200304A']:
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
            order = np.argsort(peaks.values)
            spikes.set_info(order=order, peaks=peaks)

            try:
                maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
                
            except:
                meanwf, maxch = data.load_mean_waveforms(spike_count=1000)
                maxch.to_csv(data.nwb_path + "/maxch.csv")        

            spikes.set_info(maxch = maxch[tokeep])

            if len(tokeep) > 5:
                
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
                                
                window_size = bin_size*100.0

                glm = ConvolvedGLM(spikes, bin_size, window_size, newwake_ep)
                glm.fit_scipy()            

                spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(wake_ep))
                # spikes2 = nap.randomize.resample_timestamps(spikes.restrict(wake_ep))
                spikes2.set_info(maxch = spikes._metadata["maxch"], group = spikes._metadata["group"])
                glms = ConvolvedGLM(spikes2, bin_size, window_size, newwake_ep)
                glms.fit_scipy()            

                # glm0 = ConvolvedGLM(spikes, bin_size, 1.0, newwake_ep)
                # glm0.W = np.zeros_like(glm.W)

                hmm = GLM_HMM((glm, glms))

                hmm.fit_transition(spikes, sws_ep, 0.01)

                # hmm.fit_observation(spikes, sws_ep, bin_size)
                
                # figure()
                # ax = subplot(311)        
                # plot(hmm.Z)            
                # subplot(312, sharex=ax)
                # plot(spikes.restrict(sws_ep).to_tsd("order"), '|', markersize=20)
                # subplot(313, sharex=ax)
                # plot(hmm.time_idx, hmm.O)
                # show()
                
                ############################################################################################### 
                # GLM CORRELATION
                ############################################################################################### 
                # corrglm = CorrelationGLM(spikes, data.basename)

                # cc_glm = {'wak':corrglm.fit(newwake_ep, 0.3, 3.0)[0]}

                # eps = hmm.eps            
                # for i in range(len(eps)):
                #     cc_glm['ep'+str(i)] = corrglm.fit(eps[i], 0.01, 0.5)[0]

                # rglm = pd.DataFrame(index = cc_glm['wak'].columns, columns = cc_glm.keys())
                # for e in cc_glm.keys():
                #     rglm[e] = cc_glm[e].loc[0]

                ############################################################################################### 
                # PEARSON CORRELATION
                ###############################################################################################        
                rates = {}
                for e, ep, binsize, std in zip(['wak', 'sws'], [newwake_ep, sws_ep], [0.3, 0.015], [1, 1]):
                    count = spikes.count(binsize, ep)
                    rate = count/binsize
                    rate = rate.as_dataframe()
                    rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)       
                    rate = rate.apply(zscore)                    
                    rates[e] = nap.TsdFrame(rate)
                
                eps = hmm.eps            
                for i in range(len(eps)):
                    rates['ep'+str(i)] = rates['sws'].restrict(eps[i])

                _ = rates.pop("sws")

                # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
                pairs = [data.basename+"_"+i+"-"+j for i,j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
                r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

                for ep in rates.keys():
                    tmp = np.corrcoef(rates[ep].values.T)
                    if len(tmp):
                        r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

                # #######################
                # # Session correlation
                # #######################
                # tmp = pd.DataFrame(index=[data.basename], columns=range(hmm.K))
                # for i in range(hmm.K):
                #     tmp.loc[data.basename,i] = scipy.stats.pearsonr(r['wak'], r['ep'+str(i)])[0]
                # corr.append(tmp)
                
                #######################
                # SAVING
                #######################
                allr.append(r)
                # allr_glm.append(rglm)
                # durations.append(pd.DataFrame(data=[e.tot_length('s') for e in eps], columns=[data.basename]).T)
                

    allr = pd.concat(allr, 0)
    # allr_glm = pd.concat(allr_glm, 0)
    # durations = pd.concat(durations, 0)
    # corr = pd.concat(corr, 0)

    allr = allr.dropna()
    for i in range(len(eps)):
        rb.loc[bin_size,i], _ = scipy.stats.pearsonr(allr['wak'], allr['ep'+str(i)])


figure()
plot(rb)
show()
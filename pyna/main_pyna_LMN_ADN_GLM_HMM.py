# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-12 19:07:52
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
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')


SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.5
    }

allr = {'adn':[], 'lmn':[]}
allr_glm = {'adn':[], 'lmn':[]}
durations = []
corr = {'adn':[], 'lmn':[]}

for s in datasets:
# for s in ['LMN-ADN/A5043/A5043-230301A']:
# # for s in ['LMN/A1413/A1413-200918A']:
# for s in ['LMN/A1414/A1414-200929A']:
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
        idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|adn")].index.values
        spikes = spikes[idx]

        try:
            maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
            
        except:
            meanwf, maxch = data.load_mean_waveforms(spike_count=1000)
            maxch.to_csv(data.nwb_path + "/maxch.csv")        

        spikes.set_info(maxch = maxch[idx])

        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves)    
        tcurves = tuning_curves
        SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
        spikes.set_info(SI)
        spikes.set_info(max_fr = tcurves.max())

        groups = {}
        for k in ['adn', 'lmn']:
            group = spikes.getby_threshold("SI", SI_thr[k])
            group = group.getby_threshold("rate", 1.0)
            group = group.getby_threshold("max_fr", 3.0)            
            tokeep = group.index
            tcurves = tuning_curves[tokeep]
            peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
            order = np.argsort(peaks.values)
            group.set_info(order=order, peaks=peaks)
            groups[k] = group

        spikes = groups['lmn']

        if len(groups['lmn']) > 5 and len(groups['adn']) > 3:
            
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
            
            
            glm = ConvolvedGLM(spikes, bin_size, 1.0, newwake_ep)
            glm.fit_scipy()            

            spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(wake_ep))
            # spikes2 = nap.randomize.resample_timestamps(spikes.restrict(wake_ep))
            spikes2.set_info(maxch = spikes._metadata["maxch"], group = spikes._metadata["group"])
            glms = ConvolvedGLM(spikes2, bin_size, 1.0, newwake_ep)
            glms.fit_scipy()            

            glm0 = ConvolvedGLM(spikes, bin_size, 1.0, newwake_ep)
            glm0.W = np.zeros_like(glm.W)

            hmm = GLM_HMM((glm, glms))

            hmm.fit_transition(spikes, sws_ep, 0.03)

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

            eps = hmm.eps
            durations.append(pd.DataFrame(data=[e.tot_length('s') for e in eps], columns=[data.basename]).T)
            

            for k in ['adn', 'lmn']:
                spikes = groups[k]

                ############################################################################################### 
                # PEARSON CORRELATION
                ###############################################################################################
                rates = {}
                for e, ep, bin_size, std in zip(['wak', 'sws'], [newwake_ep, sws_ep], [0.3, 0.03], [1, 1]):
                    count = spikes.count(bin_size, ep)
                    rate = count/bin_size
                    rate = rate.as_dataframe()
                    rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)       
                    rate = rate.apply(zscore)                    
                    rates[e] = nap.TsdFrame(rate)
                
                
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

                #######################
                # SAVING
                #######################
                allr[k].append(r)
                # allr_glm.append(rglm)                

                #######################
                # Session correlation
                #######################
                tmp = pd.DataFrame(index=[data.basename], columns=range(hmm.K))
                for i in range(hmm.K):
                    tmp.loc[data.basename,i] = scipy.stats.pearsonr(r['wak'], r['ep'+str(i)])[0]
                corr[k].append(tmp)
                
    
durations = pd.concat(durations, 0)

for k in ['adn', 'lmn']:
    allr[k] = pd.concat(allr[k], 0)
    # allr_glm = pd.concat(allr_glm, 0)    
    corr[k] = pd.concat(corr[k], 0)

# print(scipy.stats.wilcoxon(corr[0], corr[1]))


figure()
gs = GridSpec(2, len(eps))
for j, k in enumerate(allr.keys()):
    for i in range(len(eps)):
        subplot(gs[j,i])
        plot(allr[k]['wak'], allr[k]['ep'+str(i)], 'o', color = 'red', alpha = 0.5)
        m, b = np.polyfit(allr[k]['wak'].values, allr[k]['ep'+str(i)].values, 1)
        x = np.linspace(allr[k]['wak'].min(), allr[k]['wak'].max(),5)
        plot(x, x*m + b)
        xlabel('wak')
        ylabel('ep'+str(i))
        xlim(allr[k]['wak'].min(), allr[k]['wak'].max())
        ylim(allr[k].iloc[:,1:].min().min(), allr[k].iloc[:,1:].max().max())
        r, p = scipy.stats.pearsonr(allr[k]['wak'], allr[k]['ep'+str(i)])
        title('r = '+str(np.round(r, 3)))


figure()    
gs = GridSpec(1, len(eps)+1)
subplot(gs[0,0])
tmp = durations.values
tmp = tmp/tmp.sum(1)[:,None]
plot(tmp.T, 'o', color = 'grey')
plot(tmp.mean(0), 'o-', markersize=20)

for i, k in enumerate(corr.keys()):
    subplot(gs[0,i+1])
    plot(corr[k].values.T, 'o-', color='grey')
    ylim(0, 1)
    print(k, scipy.stats.wilcoxon(corr[k][0], corr[k][1]))
show()
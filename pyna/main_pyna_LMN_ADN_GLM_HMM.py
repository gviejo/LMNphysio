# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-08-16 18:50:03
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
    'lmn':0.2,
    'psb':1.5
    }

allr = {'adn':[], 'lmn':[]}
allr_glm = {'adn':[], 'lmn':[]}
durations = []
corr = {'adn':[], 'lmn':[]}

# for s in datasets:
for s in ["LMN-ADN/A5043/A5043-230302A"]:
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
        
        groups = spikes.getby_category("location")
        for k in ['adn', 'lmn']:
            group = groups[k].getby_threshold("SI", SI_thr[k])
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
            newwake_ep = velocity.threshold(0.05).time_support.drop_short_intervals(1).merge_close_intervals(1)

            ############################################################################################### 
            # HMM GLM
            ###############################################################################################
            
            bin_size = 0.03
            window_size = bin_size*50.0
            
            ############################################
            print("fitting GLM")
            glm = ConvolvedGLM(spikes, bin_size*10, window_size*10, newwake_ep)
            glm.fit_scipy()
            

            # spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(newwake_ep))
            # spikes2 = nap.randomize.resample_timestamps(spikes.restrict(sws_ep))
            # spikes2.set_info(maxch = spikes._metadata["maxch"], group = spikes._metadata["group"])
            # rglm = ConvolvedGLM(spikes2, bin_size, window_size, newwake_ep)
            rglm = ConvolvedGLM(spikes, bin_size, window_size, sws_ep)
            rglm.fit_scipy()

            # glm0 = ConvolvedGLM(spikes, bin_size, window_size, newwake_ep)
            # glm0.W = np.zeros_like(glm.W)

            # sys.exit()

            # hmm = GLM_HMM((glm0, glm, rglm))
            hmm = GLM_HMM((glm, rglm))
            
            hmm.fit_transition(spikes, sws_ep, bin_size)
            
            figure()
            gs = GridSpec(4,1, hspace = 0)
            ax = subplot(gs[0,0])
            plot(hmm.Z)
            ylabel("state")     
            xlim(12558, 12566)
            subplot(gs[1,0], sharex=ax)
            plot(groups["adn"].restrict(sws_ep).to_tsd("peaks"), '|', markersize=10, mew=3)
            ylabel("ADN")
            subplot(gs[2,0], sharex=ax)
            plot(groups["lmn"].restrict(sws_ep).to_tsd("peaks"), '|', markersize=10, mew=3)
            ylabel("LMN")
            ylim(0, 2*np.pi)
            subplot(gs[3,0], sharex=ax)
            plot(hmm.time_idx, hmm.O[:,0:])
            ylabel("P(O)")            

            if all([len(ep)>1 for ep in hmm.eps.values()]):
                            
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
                    
                    eps = hmm.eps
                    for i in range(len(eps)):
                        rates['ep'+str(i)] = rates['sws'].restrict(eps[i])

                    # _ = rates.pop("sws")

                    # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
                    pairs = [data.basename+"_"+i+"-"+j for i,j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
                    r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

                    for ep in rates.keys():
                        tmp = np.corrcoef(rates[ep].values.T)
                        if len(tmp):
                            r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

                    to_keep = []
                    for p in r.index:
                        tmp = spikes._metadata.loc[np.array(p.split("_")[1].split("-"), dtype=np.int32), ['group', 'maxch']]
                        if tmp['group'].iloc[0] == tmp['group'].iloc[1]:
                            if tmp['maxch'].iloc[0] != tmp['maxch'].iloc[1]:
                                to_keep.append(p)
                    r = r.loc[to_keep]


                    #######################
                    # Session correlation
                    #######################
                    tmp = pd.DataFrame(index=[data.basename], columns=range(hmm.K))
                    for i in range(hmm.K):
                        tmp.loc[data.basename,i] = scipy.stats.pearsonr(r['wak'], r['ep'+str(i)])[0]
                    corr[k].append(tmp)
                    
                    #######################
                    # SAVING
                    #######################
                    allr[k].append(r)
                    # allr_glm.append(rglm)                                 

                durations.append(pd.DataFrame(data=[e.tot_length('s') for e in eps.values()], columns=[data.basename]).T)
    
durations = pd.concat(durations)

for k in ['adn', 'lmn']:
    allr[k] = pd.concat(allr[k])
    # allr_glm = pd.concat(allr_glm, 0)    
    corr[k] = pd.concat(corr[k])

# print(scipy.stats.wilcoxon(corr[0], corr[1]))


figure()
epochs = ['sws'] + ['ep'+str(i) for i in range(len(eps))]
gs = GridSpec(3, len(epochs))
for j, k in enumerate(allr.keys()):
    for i, e in enumerate(epochs):
        subplot(gs[j,i])
        plot(allr[k]['wak'], allr[k][e], 'o', color = 'red', alpha = 0.5)
        m, b = np.polyfit(allr[k]['wak'].values, allr[k][e].values, 1)
        x = np.linspace(allr[k]['wak'].min(), allr[k]['wak'].max(),5)
        plot(x, x*m + b)
        xlabel('wak')
        ylabel(e)
        xlim(allr[k]['wak'].min(), allr[k]['wak'].max())
        ylim(allr[k].iloc[:,1:].min().min(), allr[k].iloc[:,1:].max().max())
        r, p = scipy.stats.pearsonr(allr[k]['wak'], allr[k][e])
        title('r = '+str(np.round(r, 3)))

subplot(gs[2,0])
tmp = durations.values
tmp = tmp/tmp.sum(1)[:,None]
plot(tmp.T, 'o', color = 'grey')
plot(tmp.mean(0), 'o-', markersize=20)
ylim(0, 1)
title("Durations")

subplot(gs[2,1])
for i, k in enumerate(corr.keys()):
    for j, e in enumerate(corr[k].columns):
        plot(np.random.randn(len(corr[k]))*0.1+np.ones(len(corr[k]))*j, corr[k][e], 'o')
ylim(0, 1)
# xticks(np.arange(corr.shape[1]), corr.columns)

show()

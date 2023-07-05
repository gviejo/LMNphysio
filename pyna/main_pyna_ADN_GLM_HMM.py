# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-03 16:12:49
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
from GLM import HankelGLM, ConvolvedGLM



############################################################################################### 
# GENERAL infos
###############################################################################################
# data_directory = '/mnt/DataRAID2/'
data_directory = '/mnt/ceph/users/gviejo'
# data_directory = '/media/guillaume/LaCie'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#')


SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.5
    }

allr = []
allr_glm = []

corr = []

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
        idx = spikes._metadata[spikes._metadata["location"].str.contains("adn")].index.values
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

        spikes = spikes.getby_threshold("SI", SI_thr["adn"])
        spikes = spikes.getby_threshold("rate", 1.0)
        spikes = spikes.getby_threshold("max_fr", 3.0)

        tokeep = spikes.index
        tcurves = tcurves[tokeep]
        peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        order = np.argsort(peaks.values)
        spikes.set_info(order=order, peaks=peaks)


        if len(tokeep) > 8:
            
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
            
            bin_size = 0.04
            
            
            glm = ConvolvedGLM(spikes, bin_size, 1.0, newwake_ep)
            glm.fit_scipy()

            spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(wake_ep))
            glms = ConvolvedGLM(spikes2, bin_size, 1.0, newwake_ep)
            glms.fit_scipy()


            hmm = GLM_HMM((glm, glms))

            hmm.fit(spikes, sws_ep, bin_size)

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
            '''
            pairs = []
            for n in spikes.index:
                for k in list(set(spikes.index)-set([n])):
                    pairs.append(data.basename+'_'+str(n)+'-'+str(k))
            r_glm = pd.DataFrame(index = pairs, dtype = np.float32)

            # Fit wake glm
            count = spikes.count(0.3, newwake_ep)
            Y = count.values            
            X = gaussian_filter1d(count.values.astype("float"), sigma=1, order=0)
            X = X - X.mean(0)
            X = X / X.std(0)
            N = len(spikes)
            W = []
            for n in range(N):
                model= PoissonRegressor()
                model.fit(X[:,list(set(np.arange(N))-set([n]))], Y[:,n])
                W.append(model.coef_)
            W = np.array(W)

            r_glm['wake'] = W.T.flatten()

            for i in range(n_state):
                r_glm['ep'+str(i)] = hmm.W[i].T.flatten()
            '''

            ############################################################################################### 
            # PEARSON CORRELATION
            ###############################################################################################        
            rates = {}
            for e, ep, bin_size, std in zip(['wake', 'sws'], [newwake_ep, sws_ep], [0.3, 0.02], [1, 1]):
                count = spikes.count(bin_size, ep)
                rate = count/bin_size
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
                r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

            #######################
            # Session correlation
            #######################
            tmp = pd.DataFrame(index=[data.basename], columns=range(hmm.K))
            for i in range(hmm.K):
                tmp.loc[data.basename,i] = scipy.stats.pearsonr(r['wake'], r['ep'+str(i)])[0]
            corr.append(tmp)            

            #######################
            # SAVING
            #######################
            allr.append(r)
            # allr_glm.append(r_glm)

allr = pd.concat(allr, 0)
# allr_glm = pd.concat(allr_glm, 0)

corr = pd.concat(corr, 0)


figure()

for i in range(len(eps)):
    subplot(1,len(eps),i+1)
    plot(allr['wake'], allr['ep'+str(i)], 'o', color = 'red', alpha = 0.5)
    m, b = np.polyfit(allr['wake'].values, allr['ep'+str(i)].values, 1)
    x = np.linspace(allr['wake'].min(), allr['wake'].max(),5)
    plot(x, x*m + b)
    xlabel('wake')
    ylabel('ep'+str(i))
    xlim(-1, 1)
    ylim(-1, 1)
    r, p = scipy.stats.pearsonr(allr['wake'], allr['ep'+str(i)])
    title('r = '+str(np.round(r, 3)))

show()
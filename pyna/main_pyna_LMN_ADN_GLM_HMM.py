# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   gviejo
# @Last Modified time: 2023-06-26 09:25:04
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
# data_directory = '/mnt/ceph/users/gviejo'
data_directory = '/media/guillaume/LaCie'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')


SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.5
    }

allr = []
allr_glm = []

corr = []

for s in datasets:
# # for s in ['LMN-ADN/A5002/A5002-200304A']:
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
        idx = spikes._metadata[spikes._metadata["location"].str.contains("adn|lmn")].index.values
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

        groups = {}        
        for k in ['adn', 'lmn']:
            groups[k] = spikes.getby_category("location")[k]
            groups[k] = groups[k].getby_threshold("SI", SI_thr[k])
            groups[k] = groups[k].getby_threshold("rate", 1.0)
            groups[k] = groups[k].getby_threshold("max_fr", 3.0)            
            tokeep = groups[k].index
            tmp = tcurves[tokeep]
            peaks = pd.Series(index=tmp.columns,data = np.array([circmean(tmp.index.values, tmp[i].values) for i in tmp.columns]))
            order = np.argsort(peaks.values)
            groups[k].set_info(order=order, peaks=peaks)


        if len(groups['adn'])>6 and len(groups['lmn'])>6:
            
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
            
            bin_size = 0.02
            # nbasis = 6
            # nt = 100
            n_state  = 2

            # Ws = []
            # W2s = []

            # sys.exit()

            # # GLM 0            

            # W, W2 = fit_pop_glm(spikes, newwake_ep, bin_size*1, nt, nbasis)
            # Ws.append(W)
            # W2s.append(W2)

            # # GLM 1

            # spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(wake_ep))
            # W, W2 = fit_pop_glm(spikes2, newwake_ep, bin_size*1, nt, nbasis)
            # Ws.append(W)
            # W2s.append(W2)


            # figure()
            # gs = GridSpec(2, nbasis)
            # for i in range(2):
            #     for j in range(nbasis):
            #         subplot(gs[i, j])
            #         tmp = pd.DataFrame(data=W2s[i][:,j,:], index = spikes.keys(), columns=spikes.keys())
            #         imshow(tmp.loc[order,order], aspect='auto')
            # show()

            hmm = GLM_HMM(n_state)#, nt, nbasis)            

            # sys.exit()
            spikes = groups['lmn']

            hmm.fit(spikes, sws_ep, bin_size)

            time_idx = spikes[spikes.index[0]].count(bin_size, sws_ep).index.values

            # sys.exit()

            # figure()
            # ax = subplot(311)        
            # plot(time_idx, hmm.bestZ)
            # # plot(position['ry'].restrict(newwake_ep))
            # subplot(312, sharex=ax)
            # plot(spikes.restrict(sws_ep).to_tsd("order"), '|', markersize=10)
            # subplot(313, sharex=ax)
            # plot(time_idx, hmm.O)
            # show()


            # Finding sub epochs
            Z = nap.Tsd(t=time_idx, d=hmm.bestZ, time_support=sws_ep)
            eps = []
            for i in range(n_state):
                ep = Z.threshold(i-0.5).threshold(i+0.5, "below").time_support.drop_short_intervals(0.05)
                eps.append(ep)

            sys.exit()
            ############################################################################################### 
            # GLM CORRELATION
            ###############################################################################################        
            # pairs = []
            # for n in spikes.index:
            #     for k in list(set(spikes.index)-set([n])):
            #         pairs.append(data.basename+'_'+str(n)+'-'+str(k))
            # r_glm = pd.DataFrame(index = pairs, dtype = np.float32)

            # # Fit wake glm
            # count = spikes.count(0.3, newwake_ep)
            # Y = count.values            
            # X = gaussian_filter1d(count.values.astype("float"), sigma=1, order=0)
            # X = X - X.mean(0)
            # X = X / X.std(0)
            # N = len(spikes)
            # W = []
            # for n in range(N):
            #     model= PoissonRegressor()
            #     model.fit(X[:,list(set(np.arange(N))-set([n]))], Y[:,n])
            #     W.append(model.coef_)
            # W = np.array(W)

            # r_glm['wake'] = W.T.flatten()

            # for i in range(n_state):
            #     r_glm['ep'+str(i)] = hmm.W[i].T.flatten()

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
            tmp = pd.DataFrame(index=[data.basename], columns=range(n_state))
            for i in range(n_state):
                tmp.loc[data.basename,i] = scipy.stats.pearsonr(r['wake'], r['ep'+str(i)])[0]
            corr.append(tmp)            

            #######################
            # SAVING
            #######################
            allr.append(r)
            allr_glm.append(r_glm)

allr = pd.concat(allr, 0)
allr_glm = pd.concat(allr_glm, 0)

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
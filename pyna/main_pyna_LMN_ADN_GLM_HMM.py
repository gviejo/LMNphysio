# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-05-31 14:12:16
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
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

sta_r = {'adn':[], 'lmn':[]} # trigger average of proba from the other structure spikes
cc_down = {'adn':[], 'lmn':[]} # cross corr of adn and lmn / adn down states
sta_r_down = {'adn':[], 'lmn':[]} # proba trigger on down states



# for s in datasets:
for s in ['LMN-ADN/A5043/A5043-230301A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
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
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    spikes.set_info(SI, peaks=peaks)

    adn = list(spikes.getby_category("location")["adn"].getby_threshold("SI", 0.3).index)
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

    tokeep = adn+lmn
    tokeep = np.array(tokeep)
    spikes = spikes[tokeep]    

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.001).time_support 

    ############################################################################################### 
    # HMM GLM
    ###############################################################################################
        
    groups = spikes.getby_category("location")

    if len(groups['adn'])>6 and len(groups['lmn'])>8:

        

        for g in groups.keys():

            Ws = []
            W2s = []
            
            W, W2 = fit_pop_glm(groups[g], wake_ep, 0.1, 100, 5)
            Ws.append(W)
            W2s.append(W2)

            spikes2 = nap.randomize.shuffle_ts_intervals(groups[g].restrict(wake_ep))
            W, W2 = fit_pop_glm(spikes2, wake_ep, 0.1, 100, 5)
            Ws.append(W)
            W2s.append(W2)


            hmm = GLM_HMM(Ws, 100, 5)

            hmm.fit(groups[g], sws_ep.loc[[0]], 0.01)

            sys.exit()


            ### SAVING ####
            # sta_r[g].append(sta_neurons) # trigger average of reactivtion from the other structure spikes
            # cc_down[g].append(cc_d) # cross corr of adn and lmn / adn down states
            # sta_r_down[g].append(sta_down) # reactivation trigger on down states

for i, g in enumerate(['adn', 'lmn']):
    sta_r[g] = pd.concat(sta_r[g], 1)
    sta_r_down[g] = pd.concat(sta_r_down[g], 1)
    cc_down[g] = pd.concat(cc_down[g], 1)

# sys.exit()


figure()
subplot(1,3,1)
plot(sta_r['adn'].mean(1), label = 'adn r')
plot(sta_r['lmn'].mean(1), label = 'lmn r')
legend()
xlabel("Time from the other")
title("STA proba attractorness")

subplot(1,3,2)
plot(sta_r_down['adn'].mean(1), label = 'adn r')
plot(sta_r_down['lmn'].mean(1), label = 'lmn r')
xlabel("Time from ADN down center")
title("STA proba attractorness")
legend()

subplot(1,3,3)
plot(cc_down['adn'].mean(1), label = 'adn')
plot(cc_down['lmn'].mean(1), label = 'lmn')
legend()
xlabel("Time from ADN down center")
title("CC")


figure()
subplot(2,2,1)
plot(sta_r['adn'], label = 'adn r')
title("STA proba attractorness")
xlabel("Time from LMN spikes")
legend()

subplot(2,2,3)
plot(sta_r['lmn'], label = 'lmn r')
legend()
xlabel("Time from ADN spikes")


subplot(2,2,2)
plot(sta_r_down['adn'].mean(1), label = 'adn r')
title("STA proba attractorness")

subplot(2,2,4)
plot(sta_r_down['lmn'].mean(1), label = 'lmn r')
xlabel("Time from ADN down center")

show()

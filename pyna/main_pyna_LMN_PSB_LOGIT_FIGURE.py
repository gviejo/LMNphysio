# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-08 17:54:44
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

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')



Pwak = {'psb':{}, 'lmn':{}} # trigger average of proba from the other structure spikes
Pshu = {'psb':{}, 'lmn':{}} # cross corr of psb and lmn / psb down states
RGB = {'psb':{}, 'lmn':{}} # proba trigger on down states

Pxgb = {'psb':{}, 'lmn':{}}


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
    idx = spikes._metadata[spikes._metadata["location"].str.contains("psb|lmn")].index.values
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

    psb = list(spikes.getby_category("location")["psb"].getby_threshold("SI", 0.5).index)
    lmn = list(spikes.getby_category("location")["lmn"].getby_threshold("SI", 0.1).index)

    tokeep = psb+lmn
    tokeep = np.array(tokeep)
    spikes = spikes[tokeep]    

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.001).time_support 

    ############################################################################################### 
    # LOGIT
    ###############################################################################################
    groups = spikes.getby_category("location")

    # if len(groups['psb'])>5 and len(groups['lmn'])>5:
    if len(groups['lmn'])>5:

        ## MUA ########
        mua = {
            0:nap.Ts(t=np.sort(np.hstack([groups['psb'][j].index.values for j in groups['psb'].index]))),
            1:nap.Ts(t=np.sort(np.hstack([groups['lmn'][j].index.values for j in groups['lmn'].index])))}

        mua = nap.TsGroup(mua, time_support = spikes.time_support)

        ## DOWN CENTER ######
        down_center = (down_ep["start"] + (down_ep['end'] - down_ep['start'])/2).values
        down_center = nap.TsGroup({
            0:nap.Ts(t=down_center, time_support = sws_ep)
            })

        ## SHUFFLING #####
        bin_size_wake = 0.06
        bin_size_sws = 0.02

        gmap = {'psb':'lmn', 'lmn':'psb'}

        for i, g in enumerate(gmap.keys()):

            #  WAKE 
            count = groups[g].count(bin_size_wake, newwake_ep)
            rate = count/bin_size_wake
            rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
            rate_wak = StandardScaler().fit_transform(rate)

            #  WAKE SHUFFLE
            count = nap.randomize.shuffle_ts_intervals(groups[g]).count(bin_size_wake, newwake_ep)
            rate = count/bin_size_wake
            rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
            rate_shu = StandardScaler().fit_transform(rate)
            
            # SWS
            count = groups[g].count(bin_size_sws, sws_ep)
            time_index = count.index.values
            rate = count/bin_size_sws
            rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            rate_sws = StandardScaler().fit_transform(rate)

            X = np.vstack((rate_wak, rate_shu))
            y = np.hstack((np.zeros(len(rate_wak)), np.ones(len(rate_shu)))).astype(np.int32)
            Xt = rate_sws

            # Classifier
            bst = XGBClassifier(
                n_estimators=10, max_depth=20, 
                learning_rate=0.0001, objective='binary:logistic', 
                random_state = 0,# booster='dart',
                #eval_metric=f1_score)
                )
            bst.fit(X, y)            
            # tmp = bst.predict_proba(Xt)[:,0]            
            Pxgb[g][s] = bst.predict(X)

            # PROJECTION            
            rgb = getRGB(position['ry'], newwake_ep, bin_size_wake)
            p_wak = KernelPCA(2, kernel='cosine').fit_transform(rate_wak)
            p_shu = KernelPCA(2, kernel='cosine').fit_transform(rate_shu)
            # tmp = Isomap(n_neighbors=10).fit_transform(rate_wak)

            ### SAVING ####
            Pwak[g][s] = p_wak
            Pshu[g][s] = p_shu
            RGB[g][s] = rgb




figure()
gs = GridSpec(len(Pwak["psb"]), 8)

for i, g in enumerate(['psb', 'lmn']):
    
    for j, s in enumerate(Pwak[g].keys()):

        subplot(gs[j,i*4])
        tmp = np.histogram2d(Pwak[g][s][:,0], Pwak[g][s][:,1], 20)
        imshow(tmp[0])
        # scatter(Pwak[g][s][:,0], Pwak[g][s][:,1], color = RGB[g][s])

    for j, s in enumerate(Pshu[g].keys()):

        subplot(gs[j,i*4+1])

        tmp = np.histogram2d(Pshu[g][s][:,0], Pshu[g][s][:,1], 20)
        imshow(tmp[0])

    for j, s in enumerate(Pwak[g].keys()):

        # tmp = np.vstack((Pwak[g][s], Pshu[g][s]))
        tmp = Pwak[g][s][Pxgb[g][s][0:len(Pwak[g][s])]==0]
        tmp2 = np.histogram2d(tmp[:,0], tmp[:,1], 20)

        subplot(gs[j,i*4+2])
        
        # scatter(tmp[:,0], tmp[:,1], c = Pxgb[g][s][0:len(tmp)], marker = '.', alpha = 0.2, edgecolor = None)
        imshow(tmp2[0])

    for j, s in enumerate(Pshu[g].keys()):

        # tmp = np.vstack((Pwak[g][s], Pshu[g][s]))
        tmp = Pshu[g][s][Pxgb[g][s][len(Pshu[g][s]):]==1]
        tmp2 = np.histogram2d(tmp[:,0], tmp[:,1], 20)


        subplot(gs[j,i*4+3])
        
        # scatter(tmp[:,0], tmp[:,1], c = Pxgb[g][s][len(tmp):], marker = '.', alpha = 0.2, edgecolor = None)
        imshow(tmp2[0])


figure()

gs = GridSpec(len(Pxgb["psb"]), 2)

for i, g in enumerate(['psb', 'lmn']):
    
    for j, s in enumerate(Pxgb[g].keys()):

        subplot(gs[j,i])
        plot(Pxgb[g][s])




show()


# sys.exit()



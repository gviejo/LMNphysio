# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-12-22 17:19:39
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-23 14:43:10
#!/usr/bin/env python
'''

'''
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore



############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

sta_r = {'psb':[], 'lmn':[]} # trigger average of reactivtion from the other structure spikes
cc_down = {'psb':[], 'lmn':[]} # cross corr of psb and lmn / psb down states
sta_r_down = {'psb':[], 'lmn':[]} # reactivation trigger on down states


for s in datasets:    
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

    psb = list(spikes.getby_category("location")["psb"].getby_threshold("SI", 0.06).index)
    lmn = list(spikes.getby_category("location")["lmn"].getby_threshold("SI", 0.1).index)
    
    tokeep = psb+lmn
    tokeep = np.array(tokeep)
    spikes = spikes[tokeep]    

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.001).time_support 

    ############################################################################################### 
    # REACTOvATOPM
    ############################################################################################### 
    groups = spikes.getby_category("location")

    if len(groups['psb'])>5 and len(groups['lmn'])>5:

        ## MUA ########
        mua = {
            0:nap.Ts(t=np.sort(np.hstack([groups['lmn'][j].index.values for j in groups['lmn'].index]))),
            1:nap.Ts(t=np.sort(np.hstack([groups['psb'][j].index.values for j in groups['psb'].index])))}

        mua = nap.TsGroup(mua, time_support = spikes.time_support)

        ## DOWN CENTER ######
        down_center = (down_ep["start"] + (down_ep['end'] - down_ep['start'])/2).values
        down_center = nap.TsGroup({
            0:nap.Ts(t=down_center, time_support = sws_ep)
            })

        ## REACTIVATION ####
        bin_size_wake = 0.3
        bin_size_sws = 0.02

        gmap = {'psb':'lmn', 'lmn':'psb'}

        for i, g in enumerate(gmap.keys()):
            #  WAKE 
            count = groups[g].count(bin_size_wake, newwake_ep)
            rate = count/bin_size_wake
            #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            rate = zscore_rate(rate)
            C = (1/rate.shape[0])*np.dot(rate.values.T, rate.values)
            C[np.diag_indices_from(C)] = 0.0

            # SWS
            count = groups[g].count(bin_size_sws, sws_ep)
            rate = count/bin_size_sws
            #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            rate = zscore_rate(rate)

            p = np.sum(np.dot(rate.values, C) * rate.values, 1)
            p = nap.Tsd(t=count.index.values, d = p, time_support = sws_ep)
            
            # STA / neurons        
            sta_neurons = nap.compute_event_trigger_average(groups[gmap[g]], p, bin_size_sws, (-0.4, 0.4), sws_ep)
            # sta = nap.compute_event_trigger_average(mua[[0]], p, 0.02, (-0.4, 0.4), sws_ep)        
            sta_neurons = sta_neurons.as_dataframe().apply(zscore)        
            
            # STA / down
            sta_down = nap.compute_event_trigger_average(down_center, p, bin_size_sws, (-0.4, 0.4), sws_ep)
            sta_down = sta_down.as_dataframe().apply(zscore)        

            # CC / down
            cc_d = nap.compute_eventcorrelogram(groups[g], down_center[0], bin_size_sws, 0.4, ep=sws_ep)

            ### SAVING ####
            sta_r[g].append(sta_neurons) # trigger average of reactivtion from the other structure spikes
            cc_down[g].append(cc_d) # cross corr of psb and lmn / psb down states
            sta_r_down[g].append(sta_down) # reactivation trigger on down states


for i, g in enumerate(['psb', 'lmn']):
    sta_r[g] = pd.concat(sta_r[g], 1)
    sta_r_down[g] = pd.concat(sta_r_down[g], 1)
    cc_down[g] = pd.concat(cc_down[g], 1)

# sys.exit()


figure()
subplot(1,3,1)
plot(sta_r['psb'].mean(1), label = 'psb r')
plot(sta_r['lmn'].mean(1), label = 'lmn r')
legend()
xlabel("Time from the other")
title("STA reactivation")

subplot(1,3,2)
plot(sta_r_down['psb'].mean(1), label = 'psb r')
plot(sta_r_down['lmn'].mean(1), label = 'lmn r')
xlabel("Time from psb down center")
title("STA reactivation")
legend()

subplot(1,3,3)
plot(cc_down['psb'].mean(1), label = 'psb')
plot(cc_down['lmn'].mean(1), label = 'lmn')
legend()
xlabel("Time from psb down center")
title("CC")
show()
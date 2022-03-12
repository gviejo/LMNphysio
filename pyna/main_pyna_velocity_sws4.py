# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-04 16:38:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-04 18:05:12
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
from functions import *
import sys
from itertools import combinations, product
from umap import UMAP
from matplotlib.pyplot import *
from sklearn.manifold import Isomap

data_directory = '/mnt/DataGuillaume/'

sessions = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

alltc_sws = {'adn':{}, 'lmn':{}}
alltc_wak = {'adn':{}, 'lmn':{}}

# for se in sessions:
for se in ['LMN-ADN/A5011/A5011-201014A']:
    path = os.path.join(data_directory, se)
    data = nap.load_session(path, 'neurosuite')

    spikes = data.spikes.getby_threshold('freq', 1.0)
    angle = data.position['ry']
    wake_ep = data.epochs['wake']
    sleep_ep = data.epochs['sleep']
    sws_ep = data.read_neuroscope_intervals('sws')

    idx = spikes._metadata[spikes._metadata["location"].str.contains("adn|lmn")].index.values
    spikes = spikes[idx]

    # # COMPUTING TUNING CURVES
    tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    spikes.set_info(SI=SI)
    spikes = spikes.getby_threshold('SI', 0.1, op = '>')
    tuning_curves = tuning_curves[spikes.keys()]
        
    # DECODING XGB
    # Binning Wake
    bin_size_wake = 0.1
    count = spikes.count(bin_size_wake, angle.time_support.loc[[0]])
    count = count.as_dataframe()
    ratewak = count/bin_size_wake    
    ratewak = ratewak.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
    ratewak = nap.TsdFrame(ratewak, time_support = angle.time_support.loc[[0]])
    ratewak = zscore_rate(ratewak)
    try:
        velocity = computeLinearVelocity(data.position[['x', 'z']], data.position.time_support.loc[[0]], bin_size_wake)
        newwake_ep = velocity.threshold(0.001).time_support.merge_close_intervals(5)
    except:
        velocity = computeAngularVelocity(data.position['ry'], data.position.time_support.loc[[0]], bin_size_wake)
        newwake_ep = velocity.threshold(0.07).time_support.merge_close_intervals(5)
                
    ratewak = ratewak.restrict(newwake_ep)
    angle2 = getBinnedAngle(angle, angle.time_support.loc[[0]], bin_size_wake).restrict(newwake_ep)

    # Binning sws
    bin_size_sws = 0.04
    count = spikes.count(bin_size_sws, sws_ep)
    count = count.as_dataframe()    
    ratesws = count/bin_size_wake
    ratesws = ratesws.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
    ratesws = nap.TsdFrame(ratesws, time_support = sws_ep)
    ratesws = zscore_rate(ratesws)

    sws_angle2, proba, bst = xgb_decodage(Xr=ratewak, Yr=angle2, Xt=ratesws)
    # sws_angle2 = smoothAngle(sws_angle, 1)
    av_sws = getAngularVelocity(sws_angle2, bin_size_sws)           
    logl = np.log(proba.max(1).astype(np.float64))    
    logl = logl.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
    logl = nap.Tsd(logl, time_support = sws_ep)    
    thr = np.percentile(logl, 50)
    av_sws = av_sws.restrict(logl.time_support)    

    adn = np.array(spikes.keys())[(spikes._metadata["location"] == "adn").values]
    lmn = np.array(spikes.keys())[(spikes._metadata["location"] == "lmn").values]

    peaks = tuning_curves.idxmax()

    tc_av_sws = nap.compute_1d_tuning_curves(spikes, av_sws, 20, minmax=(0,2*np.pi), ep = av_sws.time_support)          
    tc_av_sws = tc_av_sws.loc[0:np.pi]
    tc_av_sws = tc_av_sws - tc_av_sws.mean(0)
    tc_av_sws = tc_av_sws / tc_av_sws.std(0)



    figure()
    ax = subplot(411)
    for n in adn:
        plot(spikes[n].restrict(sws_ep).as_units('s').fillna(peaks.loc[n]), '|')    
    plot(sws_angle2.restrict(sws_ep).as_units('s')) 
    subplot(412, sharex = ax)
    for n in lmn:
        plot(spikes[n].restrict(sws_ep).as_units('s').fillna(peaks.loc[n]), '|')    
    plot(sws_angle2.restrict(sws_ep).as_units('s')) 
    subplot(413,sharex = ax)
    plot(av_sws)
    subplot(414, sharex = ax)
    plot(logl)
    axhline(thr)
    tight_layout()
    

    figure()
    subplot(211)
    plot(tc_av_sws[adn])
    subplot(212)
    plot(tc_av_sws[lmn])
    show()
    sys.exit()

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-08 15:12:21
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-10 10:20:23
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter



############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')


allpisi_wak = []
allpisi_sws = []


for s in datasets:    
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    angle = position['ry']
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    idx = spikes._metadata[spikes._metadata["location"].str.contains("adn|lmn")].index.values
    spikes = spikes[idx]

    ###############################################################################################
    # COMPUTING TUNING CURVES
    ###############################################################################################     
    tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    spikes.set_info(SI=SI)
    spikes = spikes.getby_threshold('SI', 0.1, op = '>')
    tuning_curves = tuning_curves[spikes.keys()]

    adn = spikes._metadata[spikes._metadata["location"].str.contains("adn")].index.values
    lmn = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values

    
    if len(spikes)>=10:
        print(s)
        # ##########################################################
        # # SWS ISI HD
        # ##########################################################        
        #spikes = spikes[adn]
        bin_size_wake = 0.3
        count = spikes.count(bin_size_wake, angle.time_support.loc[[0]])
        count = count.as_dataframe()
        ratewak = count/bin_size_wake
        # ratewak = np.sqrt(count/bin_size_wake)
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
        bin_size_sws = 0.02
        count = spikes.count(bin_size_sws, sws_ep)        
        sumcount = nap.Tsd(count.sum(1), time_support = sws_ep)
        newsws_ep = sumcount.threshold(0.5).time_support
        ratesws = count/bin_size_wake
        ratesws = ratesws.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        ratesws = nap.TsdFrame(ratesws, time_support = sws_ep)
        ratesws = zscore_rate(ratesws)
        ratesws = ratesws.restrict(newsws_ep)

        sws_angle, proba, bst = xgb_decodage(Xr=ratewak, Yr=angle2, Xt=ratesws)
        tmp = pd.Series(index = sumcount.index.values, data = np.random.uniform(0, 2*np.pi, len(sumcount)))
        tmp.loc[sws_angle.index] = sws_angle.values
        tmp = nap.Tsd(tmp, time_support = sws_ep)
        sws_angle2 = smoothAngle(tmp, 2)

        pisi_sws, xbins, ybins = compute_ISI_HD(spikes[lmn], sws_angle2, sws_ep, bins = np.geomspace(0.001, 30.0, 100))
        pisi_sws = np.array([pisi_sws[n].values for n in pisi_sws.keys()])
        
        #sys.exit()

        ########################################################
        # Saving
        ########################################################        
        allpisi_sws.append(pisi_sws)


allpisi_sws = np.vstack(allpisi_sws)


figure()

extents = [xbins[0], xbins[-1], ybins[-1], ybins[0]]
tmp = gaussian_filter(allpisi_sws.mean(0), sigma=1)
imshow(tmp, aspect = 'auto', cmap = 'jet', extent = extents)


show()

sys.exit()


tcurves = tuning_curves
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
figure()
ax = subplot(211)
for n in adn:
    plot(spikes[n].restrict(sws_ep).as_units('s').fillna(peaks.loc[n]), '|')    
plot(sws_angle2.restrict(newsws_ep).as_units('s'), '.-') 

subplot(212, sharex = ax)
for n in lmn:
    plot(spikes[n].restrict(sws_ep).as_units('s').fillna(peaks.loc[n]), '|')    
plot(sws_angle2.restrict(newsws_ep).as_units('s'), '.-') 
show()
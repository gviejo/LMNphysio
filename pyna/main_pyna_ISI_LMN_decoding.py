# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-07 18:43:39
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-12 15:00:57
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
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

allpisi_wak = []
allpisi_sws = []

tcurves_wak = []
tcurves_sws = []

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
    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
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

    ##########################################################
    # WAK ISI HD
    ##########################################################
    try:
        velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.003).time_support.merge_close_intervals(5)
    except:
        velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.05).time_support.merge_close_intervals(5)

    # ep = newwake_ep
    bins = np.geomspace(0.002, 30.0, 100)
    

    pisi_wak, xbins, ybins = compute_ISI_HD(spikes, angle, newwake_ep, bins = bins)
    pisi_wak = np.array([pisi_wak[n].values for n in pisi_wak.keys()])
    
    tcurves_wak.append(tuning_curves)    
    
    if len(spikes)>=10:
        print(s)

        ##########################################################
        # SWS ISI HD
        ##########################################################        
        bin_size_wake = 0.3
        count = spikes.count(bin_size_wake, angle.time_support.loc[[0]])
        count = count.as_dataframe()
        ratewak = count/bin_size_wake
        # ratewak = np.sqrt(count/bin_size_wake)
        ratewak = ratewak.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        ratewak = nap.TsdFrame(ratewak, time_support = angle.time_support.loc[[0]])
        ratewak = zscore_rate(ratewak)
        ratewak = ratewak.restrict(newwake_ep)
        angle2 = getBinnedAngle(angle, angle.time_support.loc[[0]], bin_size_wake).restrict(newwake_ep)
        # Binning sws
        bin_size_sws = 0.03
        count = spikes.count(bin_size_sws, sws_ep)        
        sumcount = nap.Tsd(count.sum(1), time_support = sws_ep)
        newsws_ep = sumcount.threshold(0.5).time_support
        ratesws = count/bin_size_wake
        ratesws = ratesws.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        ratesws = nap.TsdFrame(ratesws, time_support = sws_ep)
        ratesws = zscore_rate(ratesws)
        ratesws = ratesws.restrict(newsws_ep)

        sws_angle, proba, bst = xgb_decodage(Xr=ratewak, Yr=angle2, Xt=ratesws)
        tmp = pd.Series(index = sumcount.index.values, data = np.nan)
        tmp.loc[sws_angle.index] = sws_angle.values
        tmp = tmp.fillna(method='pad').fillna(0)        
        tmp = nap.Tsd(tmp, time_support = sws_ep)
        sws_angle2 = smoothAngle(tmp, 1)
        
        bins = np.geomspace(0.002, 30.0, 100)

        pisi_sws, xbins, ybins = compute_ISI_HD(spikes, sws_angle2, sws_ep, bins = bins)
        pisi_sws = np.array([pisi_sws[n].values for n in pisi_sws.keys()])        
                

        tuning_curves = nap.compute_1d_tuning_curves(spikes, sws_angle2, 120, minmax=(0, 2*np.pi), ep = sws_ep)
        tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)

        tcurves_sws.append(tuning_curves)

        
        ########################################################
        # Saving
        ########################################################
        allpisi_wak.append(pisi_wak)
        allpisi_sws.append(pisi_sws)


allpisi_wak = np.vstack(allpisi_wak)
allpisi_sws = np.vstack(allpisi_sws)
tcurves_wak = pd.concat(tcurves_wak, 1)
tcurves_sws = pd.concat(tcurves_sws, 1)

tcurves_wak.columns = np.arange(tcurves_wak.shape[1])
tcurves_wak = centerTuningCurves(tcurves_wak)
tcurves_sws.columns = np.arange(tcurves_sws.shape[1])
tcurves_sws = centerTuningCurves(tcurves_sws)



figure()
subplot(131)
extents = [xbins[0], xbins[-1], ybins[-1], ybins[0]]
tmp = gaussian_filter(allpisi_wak.mean(0), sigma=1)
imshow(tmp, aspect = 'auto', cmap = 'jet', extent = extents)

subplot(132)
extents = [xbins[0], xbins[-1], ybins[-1], ybins[0]]
tmp = gaussian_filter(allpisi_sws.mean(0), sigma=1)
imshow(tmp, aspect = 'auto', cmap = 'jet', extent = extents)

subplot(133)
semilogx(ybins[0:-1], allpisi_sws.mean(0).sum(1), label = 'sws')
semilogx(ybins[0:-1], allpisi_wak.mean(0).sum(1), label = 'wak')

show()

datatosave = {'wak':allpisi_wak, 'sws':allpisi_sws, 'bins':bins, 'tc_wak':tcurves_wak, 'tc_sws':tcurves_sws}

cPickle.dump(datatosave, open(os.path.join('../data/', 'PISI_LMN2.pickle'), 'wb'))


# tcurves = tuning_curves
# peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
# figure()
# ax = subplot(211)
# for n in spikes.keys():
#     plot(spikes[n].restrict(newsws_ep).as_units('s').fillna(peaks.loc[n]), '|')    
# plot(sws_angle2.restrict(newsws_ep).as_units('s'), '.-') 
# subplot(212, sharex=ax)
# plot(logl)
# show()

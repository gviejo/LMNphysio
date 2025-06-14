# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-07 18:43:39
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-06-14 15:21:36
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
import sys
sys.path.append("..")
from functions import *

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from pycircstat.descriptive import mean as circmean

from scipy.optimize import curve_fit


############################################################################################### 
# GENERAL infos
###############################################################################################
if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'  

datasets = np.hstack((
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
    ))

datasets = np.unique(datasets)

allpisi_wak = []
allpisi_sws = []

tcurves_wak = []
tcurves_sws = []

allscores = []

for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    
    if os.path.exists(filepath):
        
        nwb = nap.load_file(filepath)
        
        spikes = nwb['units']
        spikes = spikes.getby_threshold("rate", 1)

        position = []
        columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        for k in columns:
            position.append(nwb[k].values)
        position = np.array(position)
        position = np.transpose(position)
        position = nap.TsdFrame(
            t=nwb['x'].t,
            d=position,
            columns=columns,
            time_support=nwb['position_time_support'])

        epochs = nwb['epochs']
        wake_ep = epochs[epochs.tags == "wake"]
        sws_ep = nwb['sws']
        rem_ep = nwb['rem']
        
        spikes = spikes[(spikes.location == "lmn") | (spikes.location == "adn")]

        if len(spikes):
    
            ############################################################################################### 
            # COMPUTING TUNING CURVES
            ###############################################################################################
            tuning_curves = nap.compute_1d_tuning_curves(
                spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                ep = position.time_support.loc[[0]])
            tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
            
            SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
            spikes.set_info(SI)

            spikes = spikes[spikes.SI>0.1]

            # CHECKING HALF EPOCHS
            wake2_ep = splitWake(position.time_support.loc[[0]])    
            tokeep2 = []
            stats2 = []
            tcurves2 = []   
            for i in range(2):
                tcurves_half = nap.compute_1d_tuning_curves(
                    spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                    ep = wake2_ep[i]
                    )
                tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

                tokeep, stat = findHDCells(tcurves_half)
                tokeep2.append(tokeep)
                stats2.append(stat)
                tcurves2.append(tcurves_half)       
            tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

            print(s, len(tokeep))

            spikes = spikes[tokeep]    

            if len(spikes):
                

                adn = spikes.index[spikes.location=="adn"]
                lmn = spikes.index[spikes.location=="lmn"]

                if len(adn) > 6 or len(lmn)>9:
                
                    tcurves         = tuning_curves[tokeep]

                    try:
                        velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                        newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
                    except:
                        velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                        newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)


                    
                    ############################################################################################### 
                    # WAKEFULNESS ISI HD
                    ###############################################################################################
                    nb_bin_hd=32

                    # ep = newwake_ep
                    bins = np.geomspace(0.002, 100.0, 200)

                    # # Interpolation
                    # angle2 = np.unwrap(position['ry']).interpolate(spikes[tokeep[0]].count(0.001, wake_ep))
                    # angle2 = angle2%(2*np.pi)
                    angle2 = position['ry']

                                        
                    pisi_wak, xbins, ybins = compute_ISI_HD(spikes[lmn], angle2, newwake_ep, bins = bins, nb_bin_hd=nb_bin_hd)
                    pisi_wak = np.array([pisi_wak[n].values for n in pisi_wak.keys()])
                    
                    tcurves_wak.append(tuning_curves[lmn])


                    
                    # bin_size_wake = 0.3
                    # count = spikes.count(bin_size_wake, position.time_support.loc[[0]])
                    # count = count.as_dataframe()
                    # ratewak = count/bin_size_wake
                    # # ratewak = np.sqrt(count/bin_size_wake)
                    # ratewak = ratewak.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
                    # ratewak = nap.TsdFrame(ratewak, time_support = position.time_support.loc[[0]])
                    # ratewak = zscore_rate(ratewak)                    
                    # ratewak = ratewak.restrict(newwake_ep)
                    # angle2 = getBinnedAngle(position['ry'], position.time_support.loc[[0]], bin_size_wake).restrict(newwake_ep)


                    # ##########################################################
                    # # SWS ISI HD
                    # ##########################################################        
                    

                    # Binning sws
                    bin_size_sws = 0.01

                    # sws_angle, P = nap.decode_1d(tcurves, spikes, sws_ep, bin_size_sws)
                    count = spikes.count(bin_size_sws).smooth(bin_size_sws*2, size_factor=20, norm=False).restrict(sws_ep)
                    sws_angle, P = nap.decode_1d(tcurves, count, sws_ep, bin_size_sws)
                    

                    # sws_angle, P = decode_xgb(spikes, newwake_ep, 0.2, sws_ep, bin_size_sws, position['ry'], std=1)
                    # sws_angle2 = smoothAngle(sws_angle, 1)


                    # # Interpolation
                    sws_angle2 = np.unwrap(sws_angle).interpolate(spikes[tokeep[0]].count(0.002, sws_ep))
                    sws_angle2 = sws_angle2%(2*np.pi)                    
                    

                    # Thresholding
                    count = spikes.count(bin_size_sws).smooth(bin_size_sws, size_factor=20, norm=False).restrict(sws_ep)
                    sumcount = count.sum(1)
                    new_sws_ep = sumcount.threshold(1.5).time_support.drop_short_intervals(0.05)

                    # ratesws = count/bin_size_wake
                    # # ratesws = ratesws.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
                    # ratesws = ratesws.smooth(std=bin_size_sws*2, size_factor=20)                
                    # ratesws = zscore_rate(ratesws)
                    # ratesws = ratesws.restrict(newsws_ep)
                    # sws_angle, proba, bst = xgb_decodage(Xr=ratewak, Yr=angle2, Xt=ratesws)
                    # tmp = pd.Series(index = sumcount.index.values, data = np.nan)
                    # tmp.loc[sws_angle.index] = sws_angle.values
                    # tmp = tmp.ffill()#fillna(method='pad').fillna(0)        
                    # tmp = nap.Tsd(tmp, time_support = sws_ep)
                    # sws_angle2 = smoothAngle(tmp, 1)
                    
                    pisi_sws, xbins, ybins = compute_ISI_HD(spikes[lmn], sws_angle2, new_sws_ep, bins = bins, nb_bin_hd=nb_bin_hd)
                    pisi_sws = np.array([pisi_sws[n].values for n in pisi_sws.keys()])
                    
                    tuning_curves = nap.compute_1d_tuning_curves(spikes[lmn], sws_angle2, 120, minmax=(0, 2*np.pi), ep = new_sws_ep)
                    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)

                    tcurves_sws.append(tuning_curves)

                    
                    ########################################################
                    # Saving
                    ########################################################
                    allpisi_wak.append(pisi_wak)
                    allpisi_sws.append(pisi_sws)



allpisi_wak = np.vstack(allpisi_wak)
allpisi_sws = np.vstack(allpisi_sws)
tcurves_wak = pd.concat(tcurves_wak, axis=1)
tcurves_sws = pd.concat(tcurves_sws, axis=1)

tcurves_wak.columns = np.arange(tcurves_wak.shape[1])
tcurves_wak = centerTuningCurves_with_peak(tcurves_wak)
tcurves_sws.columns = np.arange(tcurves_sws.shape[1])
tcurves_sws = centerTuningCurves_with_peak(tcurves_sws)





figure()
subplot(231)
extents = [xbins[0], xbins[-1], ybins[-1], ybins[0]]
tmp = allpisi_wak.mean(0)
tmp = tmp/tmp.sum(0)
tmp = gaussian_filter(tmp, sigma=1)
imshow(tmp, aspect = 'auto', cmap = 'jet', extent = extents)

subplot(232)
extents = [xbins[0], xbins[-1], ybins[-1], ybins[0]]
tmp = allpisi_sws.mean(0)
tmp = tmp/tmp.sum(0)
tmp = gaussian_filter(tmp, sigma=1)
imshow(tmp, aspect = 'auto', cmap = 'jet', extent = extents)

subplot(233)
semilogx(ybins[0:-1], allpisi_sws.mean(0).sum(1), label = 'sws')
semilogx(ybins[0:-1], allpisi_wak.mean(0).sum(1), label = 'wak')
legend()
title("LMN")


savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_isi/"+"isi_lmn.pdf"))
show()



# tcurves = tuning_curves
# peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
# figure()
# ax = subplot(111)
# for n in spikes.keys():
#     plot(spikes[n].restrict(newsws_ep).as_units('s').fillna(peaks.loc[n]), '|')    
# plot(sws_angle.restrict(newsws_ep).as_units('s'), '.-') 

datatosave = {'wak':allpisi_wak, 
        'sws':allpisi_sws, 
        'bins':bins, 
        'tc_wak':tcurves_wak, 
        'tc_sws':tcurves_sws,
        }



dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'PISI_LMN.pickle'), 'wb'))


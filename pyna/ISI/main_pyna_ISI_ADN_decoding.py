# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-07 18:43:39
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-06-14 15:43:37
# %%
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
import yaml




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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#')


datasets2 = yaml.safe_load(
    open(os.path.join(
        data_directory,
        "datasets_OPTO.yaml"), "r"))['opto']

datasets2 = ["OPTO/"+s for s in datasets2['adn']['opto']['ipsi']['sleep']]

datasets = np.hstack((datasets, datasets2))


allpisi_wak = []
allpisi_sws = []

tcurves_wak = []
tcurves_sws = []

allscores = []

for s in datasets:    
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    # filepath = os.path.join(path, "pynapplenwb", basename + ".nwb")
    
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
        sleep_ep = epochs[epochs.tags == "sleep"]
        sws_ep = nwb['sws']
        # rem_ep = nwb['rem']
        
        # data = nap.load_session(path, 'neurosuite')
        # spikes = data.spikes
        # position = data.position
        # epochs = data.epochs
        # wake_ep = epochs['wake']
        # sleep_ep = epochs['sleep']
        # sws_ep = data.read_neuroscope_intervals('sws')
        
        
        if "OPTO" in s:
            sws_ep = sws_ep.intersect(sleep_ep[0])
        

        spikes = spikes[spikes.location == "adn"]
        

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

            spikes = spikes[spikes.SI>0.3]

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

            if len(tokeep) > 5:
                print(s)

                spikes = spikes[tokeep]
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
                # angle2 = np.unwrap(position['ry']).interpolate(spikes[tokeep[0]].count(0.002, wake_ep))
                # angle2 = angle2%(2*np.pi)
                angle2 = position['ry']
                
                pisi_wak, xbins, ybins = compute_ISI_HD(spikes, angle2, newwake_ep, bins = bins, nb_bin_hd=nb_bin_hd)
                pisi_wak = np.array([pisi_wak[n].values for n in pisi_wak.keys()])
                
                tcurves_wak.append(tuning_curves)                

                # a = pisi_wak.mean(-1).T                 
                # plot(gaussian_filter1d(a, 2, axis=0)[:,0])
                # show()
                
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



                # # # Interpolation
                sws_angle2 = np.unwrap(sws_angle).interpolate(spikes[tokeep[0]].count(0.002, sws_ep))
                sws_angle2 = sws_angle2%(2*np.pi)


                # Thresholding
                count = spikes.count(bin_size_sws).smooth(bin_size_sws, size_factor=20, norm=False).restrict(sws_ep)
                sumcount = count.sum(1)
                new_sws_ep = sumcount.threshold(1.5).time_support.drop_short_intervals(0.05)

                # sumcount = sumcount/sumcount.max()
                # new_sws_ep = sumcount.threshold(np.percentile(sumcount, 0.01)).time_support
                # new_sws_ep = sws_ep

                
                pisi_sws, xbins, ybins = compute_ISI_HD(spikes, sws_angle2, new_sws_ep, bins = bins, nb_bin_hd = nb_bin_hd)
                pisi_sws = np.array([pisi_sws[n].values for n in pisi_sws.keys()])        
                
                tuning_curves = nap.compute_1d_tuning_curves(spikes, sws_angle2, 120, minmax=(0, 2*np.pi), ep = new_sws_ep)
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





######################################################
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
title("ADN")


savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_isi/"+"isi_adn.pdf"))
show()







datatosave = {'wak':allpisi_wak, 
        'sws':allpisi_sws, 
        'bins':bins, 
        'tc_wak':tcurves_wak, 
        'tc_sws':tcurves_sws
        }


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'PISI_ADN.pickle'), 'wb'))
# cPickle.dump(datatosave, open(os.path.join('../data/', 'PISI_ADN2.pickle'), 'wb'))



# ##############################################
# # FITTING SIGMOIDE
# ##############################################
# def sigmoid(x, L, x0, k):
#     return L / (1 + np.exp(-k * (x - x0)))

# def sigmoid_offset(x, A, L, x0, k):
#     return A + L / (1 + np.exp(-k * (x - x0)))     

# from scipy.optimize import OptimizeWarning
# import warnings
# from scipy.optimize import curve_fit

# Ypred = {}
# Ydata = {}
# betas = {}

# for e, pisi in zip(['wak', 'sws'], [allpisi_wak, allpisi_sws]):
#     # folding pisi in 2
#     n = (pisi.shape[-1]//2)
#     a = (pisi[:,:,0:n] + pisi[:,:,-n:][:,:,::-1])/2
#     # b = gaussian_filter(a, (0, 1, 1))
#     b = a
#     m = np.argmax(b, 1)
#     t = m.T[::-1]
#     # t = bins[m].T[::-1]
    
#     ydata = np.pad(t, ((10, 10), (0,0)), 'edge')    
#     ydata = ydata - ydata.mean(0)
#     ydata = ydata / ydata.std(0)
#     # ydata = gaussian_filter1d(ydata, sigma=3, axis=0)


#     xdata = np.linspace(-10, 10, len(ydata))

#     popt = []
#     idx = []    
#     for k in range(ydata.shape[1]):
#         with warnings.catch_warnings():
#             warnings.simplefilter("error", OptimizeWarning)        
#             try:
#                 p, _ = curve_fit(sigmoid_offset, xdata, ydata[:,k], p0=[0, 1, 0, 1])
#                 popt.append(p)
#                 idx.append(k)                
#             except:
#                 pass

#     popt = np.array(popt)
#     idx = np.array(idx)

#     ypred = np.array([sigmoid_offset(xdata, *popt[k]) for k in range(len(popt))]).T

#     Ypred[e] = ypred
#     if len(idx):
#         Ydata[e] = ydata[:,idx]
#     else:
#         Ydata[e] = ydata
#     betas[e] = popt
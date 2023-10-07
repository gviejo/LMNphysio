# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-11 19:11:21
import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *
# import pynacollada as pyna
# from scipy.signal import filtfilt
from scipy.stats import zscore
from numba import jit
from sklearn.metrics.pairwise import *

@jit(nopython=True)
def get_successive_bins(s):
    idx = np.zeros(len(s)-1)
    for i in range(len(s)-1):
        if s[i] > 1 or s[i+1] > 1:
            idx[i] = 1
    return idx


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



datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')

SI_thr = {
    'adn':0.5, 
    'lmn':0.2,
    'psb':1.5
    }


allsim = {k:[] for k in ['wak', 'rem', 'sws']}

for s in datasets:
# for s in ['LMN-PSB/A3019/A3019-220701A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes.getby_threshold('rate', 0.4)
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')

    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|psb")].index.values
    spikes = spikes[idx]

    angle = position['ry']
    tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    spikes.set_info(SI)
    r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
    spikes.set_info(halfr = r, SI = SI)

    psb = spikes.getby_category("location")['psb'].getby_threshold('halfr', 0.5).index
    lmn = spikes.getby_category("location")['lmn'].getby_threshold('halfr', 0.5).index

    tokeep = np.hstack((psb, lmn))

    # spikes = spikes.getby_category("location")['psb']
    # spikes = spikes[psb]

    # figure()
    # for i in range(len(psb)):
    #     subplot(10, 4, i+1, projection='polar')
    #     plot(tuning_curves[psb[i]])
    # show()
    # sys.exit()

    if len(psb)>5:
        ##################################################################################################
        # COUNT
        ##################################################################################################
        rates = {}
        counts = {}
        for e, ep, bin_size, std in zip(
                ['wak', 'rem', 'sws'], 
                [wake_ep, rem_ep, sws_ep], 
                [0.3, 0.3, 0.03],
                [1, 1, 1]
                ):
            count = spikes[psb].count(bin_size, ep)
            rate = count/bin_size
            rate = rate.as_dataframe()
            # rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
            rate = rate.apply(zscore)
            rates[e] = rate        
            counts[e] = count

        ##################################################################################################
        # COSINE SIMILARITY
        ##################################################################################################
        sim = {}
        for k in rates.keys():
            r = rates[k].values
            nrm = np.linalg.norm(r, axis=1)
            sim[k] = np.sum(r[0:-1]*r[1:], 1)/(nrm[0:-1]*nrm[1:])

        idxs = {}
        for k in sim.keys():
            # s = rates[k].mean(1).values
            idxs[k] = get_successive_bins(counts[k].values.sum(1))#, thr=0)
            allsim[k].append(sim[k][idxs[k] == 1])

    
for k in allsim.keys():
    allsim[k] = np.hstack(allsim[k])


figure()

for i,k in enumerate(allsim.keys()):
    subplot(1,3,i+1)
    hist(allsim[k], 100)
    xlim(-1, 1)

show()
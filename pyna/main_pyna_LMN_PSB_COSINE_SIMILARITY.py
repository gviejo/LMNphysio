# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-05-07 18:09:42
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
        if s[i] > 2 or s[i+1] > 2:
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
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data' 



datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')

SI_thr = {
    'adn':0.5, 
    'lmn':0.2,
    'psb':1.5
    }


speed_corr = {"log":{}, "lin":{}}
rate_corr = {"log":{}, "lin":{}}

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

        nwb.close()
        
        spikes = spikes[(spikes.location=="psb")|(spikes.location=="lmn")]


        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
        spikes.set_info(SI)

        # spikes = spikes[spikes.SI>0.1]


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
        
        spikes = spikes[tokeep]

        psb = spikes.index[spikes.location=="psb"]
        lmn = spikes.index[spikes.location=="lmn"]
        
        tcurves = tuning_curves[tokeep]
        # tcurves = tuning_curves



        # Filtering by SI only for LMN
        lmn = np.intersect1d(lmn[spikes.SI[lmn]>0.1], tokeep)


        try:
            velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
        except:
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)


        if len(lmn) > 4:

            ############################################################################################### 
            # PEARSON CORRELATION
            ###############################################################################################
            rates = {}
            counts = {}           
            for e, ep, bin_size, std in zip(['wak', 'sws'], 
                    [newwake_ep, sws_ep], [0.2, 0.02], [3, 3]):
                ep = ep.drop_short_intervals(bin_size*22)
                count = spikes.count(bin_size, ep)
                rate = count/bin_size
                # rate = rate.as_dataframe()
                rate = rate.smooth(std=bin_size*std, windowsize=bin_size*20).as_dataframe()
                rate = rate.apply(zscore)
                rates[e] = nap.TsdFrame(rate, time_support = ep)
                counts[e] = count

            tmp = np.corrcoef(rates['wak'].loc[lmn].values.T)
            r_wak = tmp[np.triu_indices(tmp.shape[0], 1)]


            ##################################################################################################
            # PEARSON CORRELATION AS A FUNCTION OF MUA 
            ##################################################################################################

            mua = spikes[psb].to_tsd().count(0.02, sws_ep)
            mua = mua.smooth(std=0.04, size_factor=20, norm=True)
            mua = mua/mua.max()
            
            for z, bins in zip(
                ["lin", "log"], 
                [np.linspace(0, 1, 11)[1:], np.geomspace(0.001, 1, 11)[1:]]
                ):

                R = []

                for i, m in enumerate(bins):
                    ep = mua.threshold(m, method='belowequal').time_support
                    tmp = np.corrcoef(rates['sws'].loc[lmn].restrict(ep).values.T)
                    r_sws = tmp[np.triu_indices(tmp.shape[0], 1)]

                    s, p = scipy.stats.pearsonr(r_wak, r_sws)

                    R.append(s)

                R = pd.Series(index=bins, data=R)

                rate_corr[z][s] = R

            ##################################################################################################
            # COSINE SIMILARITY
            ##################################################################################################
            bin_size = 0.02
            ep = sws_ep.drop_short_intervals(bin_size*22)
            psb_count = spikes[psb].count(0.02, ep)
            rate = psb_count/bin_size
            # rate = np.
            # rate = rate.smooth(std=bin_size*3, windowsize=bin_size*20).as_dataframe()

            rate = rate.apply(zscore)
            rate = nap.TsdFrame(rate, time_support = ep)
            nrm = np.linalg.norm(rate, axis=1)
            sim = nap.Tsd(
                t = rate.t[0:-1],
                d = np.sum(rate[0:-1].values*rate[1:].values, 1)/(nrm[0:-1].values*nrm[1:].values),
                time_support = rate.time_support
                )
            sim = sim.smooth(std=0.04, windowsize=0.02*20)
            sim = sim + 2
            
            for z, cos_bins in zip(
                ["lin", "log"], 
                [np.linspace(1, 3, 20)[1:], 4-np.geomspace(1, 3, 20)[::-1]]
                ):

                R = []

                for i, t in enumerate(cos_bins):
                    ep = sim.threshold(t, method="below").time_support

                    tmp = np.corrcoef(rates['sws'].loc[lmn].restrict(ep).values.T)
                    r_sws = tmp[np.triu_indices(tmp.shape[0], 1)]

                    s, p = scipy.stats.pearsonr(r_wak, r_sws)

                    R.append(s)

                R = pd.Series(index=cos_bins-2, data=R)

                speed_corr[z][s] = R

for z in ['lin', 'log']:
    speed_corr[z] = pd.DataFrame(speed_corr[z])
    rate_corr[z] = pd.DataFrame(rate_corr[z])

figure(figsize = (12, 12))
gs = GridSpec(2,2)
for i, (corr, func) in enumerate([('lin', plot), ('log', semilogx)]):
    subplot(gs[i,0])
    func(rate_corr[z], 'o-')
    xlabel(r"% rate")
    ylabel("Pearson r")
    ylim(0.2, 1)


    subplot(gs[i,1])
    func(speed_corr[z], 'o-')
    xlabel("Cosine similarity")
    ylabel("Pearson r")
    ylim(0.2, 1)


savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/summary_psb/fig_correlation_rate_speed.pdf")
    )


show()
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-20 18:04:17
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
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PoissonRegressor
from GLM_HMM import GLM_HMM
from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import Isomap
from mpl_toolkits import mplot3d

############################################################################################### 
# GENERAL infos
###############################################################################################
# data_directory = '/mnt/DataRAID2/'
data_directory = '/mnt/ceph/users/gviejo'
# data_directory = '/media/guillaume/LaCie'
# data_directory = '/media/guillaume/Raid2'

# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.5
    }

allr = []
allr_glm = []
durations = []
corr = []

# for s in datasets:
for s in ['LMN-ADN/A5043/A5043-230301A']:
# for s in ['LMN-PSB/A3010/A3010-210324A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    if os.path.isdir(os.path.join(path, "pynapplenwb")):

        data = nap.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake'].loc[[0]]
        sleep_ep = data.epochs['sleep'].loc[[0]]
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')
        # down_ep = data.read_neuroscope_intervals('down')

        idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
        spikes = spikes[idx]
          
        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves)    
        tcurves = tuning_curves
        SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
        spikes.set_info(SI)
        spikes.set_info(max_fr = tcurves.max())

        spikes = spikes.getby_threshold("SI", SI_thr["lmn"])
        spikes = spikes.getby_threshold("rate", 1.0)
        spikes = spikes.getby_threshold("max_fr", 3.0)

        tokeep = spikes.index
        tcurves = tcurves[tokeep]
        peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        order = np.argsort(peaks.values)
        spikes.set_info(order=order, peaks=peaks)

        try:
            maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
            
        except:
            meanwf, maxch = data.load_mean_waveforms(spike_count=100)
            maxch.to_csv(data.nwb_path + "/maxch.csv")        

        spikes.set_info(maxch = maxch[tokeep])

        if len(tokeep) > 5:
            
            # figure()
            # for i in range(len(tokeep)):
            #     subplot(4, 4, i+1, projection='polar')
            #     plot(tcurves[tokeep[i]])
            # show()
            
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1).merge_close_intervals(1)

            ############################################################################################### 
            # HMM GLM
            ###############################################################################################
            
            bin_size = 0.03
            window_size = bin_size*50.0

            sleep_ep = nap.IntervalSet(start=sleep_ep.start.values, end=sleep_ep.start.values+2000)
            sws_ep = sws_ep.intersect(sleep_ep)            

            # ############################################
            # glms = []
            # for _ in range(3):
            #     glm = ConvolvedGLM(spikes, bin_size, window_size, newwake_ep)
            #     # glm.fit_scipy()
            #     glm.W = np.zeros((glm.X.shape[-1], glm.N))
            #     glms.append(glm)
            # hmm = GLM_HMM(tuple(glms))
            # hmm.fit_observation(spikes, sws_ep, bin_size)


            ############################################
            glm = ConvolvedGLM(spikes, bin_size, window_size, newwake_ep)
            glm.fit_scipy()

            sys.exit()
            # spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(newwake_ep))
            # spikes2 = nap.randomize.resample_timestamps(spikes.restrict(sws_ep))
            # spikes2.set_info(maxch = spikes._metadata["maxch"], group = spikes._metadata["group"])
            rglm = ConvolvedGLM(spikes, bin_size, window_size, sws_ep)
            rglm.fit_scipy()

            glm0 = ConvolvedGLM(spikes, bin_size, window_size, newwake_ep)
            glm0.W = np.zeros_like(glm.W)

            hmm = GLM_HMM((glm0, glm, rglm))
            hmm.fit_transition(spikes, sws_ep, bin_size)

            # # sys.exit()
            # figure()
            # for i in range(len(spikes)):
            #     w = glm.W[:,i]
            #     a = peaks.values[list(set(np.arange(len(spikes))) - set([i]))]
            #     tmp = pd.Series(index=a, data=w)
            #     tmp = tmp.sort_index()
            #     subplot(3, 4, i+1)
            #     plot(tmp, 'o-')
            #     plot([peaks.values[i], peaks.values[i]], [0, w.max()])
            # show()

            # figure()
            # ax = subplot(311)
            # plot(hmm.Z)            
            # subplot(312, sharex=ax)
            # plot(spikes.restrict(sws_ep).to_tsd("order"), '|', markersize=20)
            # subplot(313, sharex=ax)
            # plot(hmm.time_idx, hmm.O[:,1:])
            # show()
            
            ##########################################################################################
            # Manifold
            ##########################################################################################

            count = spikes.count(bin_size, sws_ep)
            time_idx = count.index.values
            count = count.as_dataframe()
            rate = np.sqrt(count/bin_size)
            rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            rate = StandardScaler().fit_transform(rate.values)
            
            # imap = PCA(n_components=3).fit_transform(rate)

            # imap = KernelPCA(n_components=3, kernel="cosine").fit_transform(rate)
            imap = Isomap(n_components=2).fit_transform(rate)
            imap = nap.TsdFrame(t=time_idx, d=imap, time_support=sws_ep)


fig = figure()
ax = fig.add_subplot(111)

for i in range(len(hmm.eps)):
    tmp = imap.restrict(hmm.eps[i])
    ax.scatter(tmp[0], tmp[1])

show()


fig = figure()
ax = fig.add_subplot(projection='3d')

for i in range(len(hmm.eps)):
    tmp = imap.restrict(hmm.eps[0])
    ax.scatter(tmp[0], tmp[1], tmp[2], s=10)

show()

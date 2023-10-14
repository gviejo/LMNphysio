# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-10-14 17:45:52
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
import sys, os
sys.path.append("..")
from functions import *
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PoissonRegressor
from GLM_HMM import GLM_HMM
from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


'''
1. Log likelihood
2. HMMGLM vs HMMGLM intercept
3. Rasters
4. State scoring
'''


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


# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


SI_thr = {
    'adn':0.5, 
    'lmn':0.2,
    'psb':1.5
    }

allr = []
allr_glm = []
durations = []
spkcounts = []
corr = []

# for s in datasets:    
for s in ["LMN-ADN/A5026/A5026-210726A"]:
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
        
        # print(s, len(tokeep))

        # continue

        if len(tokeep) > 8:
            
            figure()
            for i in range(len(tokeep)):
                subplot(4, 4, i+1, projection='polar')
                plot(tcurves[tokeep[i]])
            
            
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = np.abs(velocity).threshold(0.02).time_support.drop_short_intervals(1).merge_close_intervals(1)

            ############################################################################################### 
            # HMM GLM
            ###############################################################################################
            
            bin_size = 0.02
            window_size = bin_size*50.0

            ############################################
            print("fitting GLM")
            glm = ConvolvedGLM(spikes, bin_size, window_size, wake_ep)
            glm.fit_scipy()
            # glm.fit_sklearn()            

            spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(wake_ep))
            # spikes2 = nap.randomize.resample_timestamps(spikes.restrict(wake_ep))
            spikes2.set_info(maxch = spikes._metadata["maxch"], group = spikes._metadata["group"])
            rglm = ConvolvedGLM(spikes2, bin_size, window_size, wake_ep)
            rglm.fit_scipy()
            # rglm.fit_sklearn()

            spikes0 = nap.TsGroup({i:nap.Ts(np.array([])) for i in spikes.keys()}, time_support=wake_ep)
            glm0 = ConvolvedGLM(spikes0, bin_size, window_size, newwake_ep)
            glm0.fit_scipy()

            glms = (glm0, glm, rglm)
            
            hmm = GLM_HMM(glms)

            ###################################################################
            # Generating new data
            ###################################################################
            K = len(glms)
            A = np.eye(K) + np.random.rand(K, K)*0.005
            A = A/A.sum(1)[:,None]

            
            t = position['ry'].bin_average(bin_size, wake_ep).index.values
            nt = len(t)
            states = np.zeros(nt, dtype="int")            
            for i in range(1, nt):
                states[i] = np.sum(np.cumsum(A[states[i-1]])<np.random.random())            
            states = nap.Tsd(t=t, d=states, time_support = wake_ep)
            
            # states to ep
            eps = []
            for i in range(len(glms)):
                eps.append(states[states.values == i].find_support(bin_size*2))

            spikes_glm = (spikes0, spikes, spikes2)

            spikes_test = {}
            for n in spikes.keys():
                tmp = []
                for j in range(len(glms)):
                    tmp.append(spikes_glm[j][n].restrict(eps[j]).index.values)
                tmp = np.sort(np.hstack(tmp))
                spikes_test[n] = nap.Ts(tmp)

            spikes_test = nap.TsGroup(spikes_test, time_support = wake_ep, peaks=spikes.get_info("peaks"))
            
            hmm.fit_transition(spikes_test, wake_ep, bin_size)

            print(np.sum(states.values == hmm.Z.values)/len(states))
                    
            figure()
            gs = GridSpec(3,1)
            ax = subplot(gs[0,0])
            plot(states, 'o', label = "True")
            plot(hmm.Z, label = "HMM")
            
            ylabel("state")
            subplot(gs[1,0], sharex=ax)
            for i in range(len(eps)):
                plot(spikes_test.restrict(eps[i]).to_tsd("peaks"), '|', markersize=20)
            ylabel("Spikes")
            ylim(0, 2*np.pi)
            subplot(gs[2,0], sharex=ax)
            plot(hmm.time_idx, hmm.O[:,0:])
            ylabel("P(O)")
            

            figure()
            [plot(s, 'o-') for s in hmm.scores]
            show()

##################################################################
# FOR FIGURE supp 1
##################################################################


datatosave = {
    "W":np.array([glms[i].W for i in range(len(glms))]),
    "scores":hmm.scores,
    "A":A,
    "bestA":hmm.A,
    "states":states,
    "bestZ":hmm.Z,
    "spikes":spikes_test,
    "tuning_curves":tcurves,
    "tokeep":tokeep,
    "peaks":peaks,
    "eps":eps,
    "order":order
    }


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
file_name = "DATA_SUPP_FIG_1_HMM_exemple.pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)



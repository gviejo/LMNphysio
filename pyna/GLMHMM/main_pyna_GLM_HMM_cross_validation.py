# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-10-01 16:36:57
# %%
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
from GLM_HMM import GLM_HMM_nemos
import nemos as nmo
import nwbmatic as ntm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# nap.nap_config.set_backend("jax")
nap.nap_config.suppress_conversion_warnings = True

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
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


SI_thr = {
    'adn':0.2, 
    'lmn':0.1,
    'psb':1.0
    }

allr = []
allr_glm = []
durations = []
spkcounts = []
corr = []

# for s in datasets:
for s in ['LMN-ADN/A5044/A5044-240401A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    if os.path.isdir(os.path.join(path, "pynapplenwb")):

        data = ntm.load_session(path, 'neurosuite')

        try:
            spikes = nap.load_file(os.path.join(path, "kilosort4/spikes_ks4.npz"))
            print("Loading KS4")
        except:
            spikes = data.spikes

        position = data.position
        wake_ep = data.epochs['wake'].loc[[0]]
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')
        # down_ep = data.read_neuroscope_intervals('down')


        idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|adn")].index.values
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

        if "adn" in spikes.location.values:
            spikes_adn = spikes[spikes.location=="adn"].getby_threshold("SI", SI_thr["adn"])
            spikes_adn = spikes_adn.getby_threshold("rate", 1.0)
            spikes_adn = spikes_adn.getby_threshold("max_fr", 3.0)
            tokeep = spikes_adn.index
            tcurves2 = tcurves[tokeep]
            peaks = pd.Series(index=tcurves2.columns,data = np.array([circmean(tcurves2.index.values, tcurves2[i].values) for i in tcurves2.columns]))
            order = np.argsort(peaks.values)
            spikes_adn.set_info(order=order, peaks=peaks)

        spikes = spikes[spikes.location=="lmn"].getby_threshold("SI", SI_thr["lmn"])
        spikes = spikes.getby_threshold("rate", 1.0)
        spikes = spikes.getby_threshold("max_fr", 3.0)
        tokeep = spikes.index
        tcurves = tcurves[tokeep]
        peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        order = np.argsort(peaks.values)
        spikes.set_info(order=order, peaks=peaks)

        # try:
        #     maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
            
        # except:
        #     meanwf, maxch = data.load_mean_waveforms(spike_count=100)
        #     maxch.to_csv(data.nwb_path + "/maxch.csv")        

        # spikes.set_info(maxch = maxch[tokeep])

        print(s, tokeep)
        if len(tokeep) > 5:
            
            # figure()
            # for i in range(len(tokeep)):
            #     subplot(4, 4, i+1, projection='polar')
            #     plot(tcurves[tokeep[i]])
            
            
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = np.abs(velocity).threshold(0.02).time_support.drop_short_intervals(1).merge_close_intervals(1)

            ############################################################################################### 
            # CROSS_VALIDATIONS
            ###############################################################################################
            
            bin_size = 0.015
            window_size = bin_size*50.0
            
            Y1 = spikes.count(bin_size, newwake_ep)
            Y2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(newwake_ep)).count(bin_size, newwake_ep)
            spikes0 = nap.TsGroup(
                {i:nap.Ts(
                    np.sort(np.random.choice(spikes[i].t, int(0.001*len(spikes[i])), replace=False))
                    ) for i in spikes.keys()}, time_support=newwake_ep)
            Y0 = spikes0.count(bin_size, newwake_ep)

            scores = []

            for i, Y in enumerate([Y0, Y1, Y2]):

                mask = np.repeat(1-np.eye(len(spikes)), 3, axis=0)

                basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs=3, shift=False, mode="conv", window_size=int(window_size/bin_size)) * Y.shape[1]

                param_grid = dict(
                    glm__regularizer_strength=(0.1, 0.01, 0.001, 1e-6),
                    # basis__n_basis_funcs=(3, 5),
                )

                pipeline = Pipeline([
                    ("basis", basis.to_transformer()),                
                    ("glm", nmo.glm.PopulationGLM(regularizer_strength=0.5, regularizer="Ridge", feature_mask=mask)),])
                    
                pipeline.fit(Y, Y)

                score = pipeline.score(Y, Y)

                scores.append(score)
                                

                gridsearch = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    cv=3
                )

                gridsearch.fit(Y.values, Y.values)

                sys.exit()




            # HMM
            ############################################
            
            hmm = GLM_HMM_nemos((glm0, glm, rglm))
            # hmm = GLM_HMM((glm, rglm))

            Y = spikes.count(bin_size, sws_ep)
            X = basis.compute_features(Y)
            
            hmm.fit_transition(X, Y)
            

            

            # #ex_ep = nap.IntervalSet(10810.81, 10847.15)
            # #(10846, 10854)

            # figure()
            # gs = GridSpec(4,1)
            # ax = subplot(gs[0,0])
            # plot(hmm.Z)
            # ylabel("state")     
            
            # subplot(gs[1,0], sharex=ax)
            # plot(spikes_adn.restrict(sws_ep).to_tsd("peaks"), '|', markersize=20)

            # subplot(gs[2,0], sharex=ax)
            # plot(spikes.restrict(sws_ep).to_tsd("peaks"), '|', markersize=20)
            # ylabel("Spikes")
            # ylim(0, 2*np.pi)
            
            # subplot(gs[3,0], sharex=ax)
            # [plot(hmm.O[:,i], label = str(i)) for i in range(3)]
            # ylabel("P(O)")
            # legend()


            # show()
            # sys.exit()
            

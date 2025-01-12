# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   gviejo
# @Last Modified time: 2025-01-05 16:16:03
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
from GLM_HMM import GLM_HMM_nemos, overlap_count
# from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
import nemos as nmo
# import nwbmatic as ntm

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')


allr = {'psb':[], 'lmn':[]}
allr_glm = {'psb':[], 'lmn':[]}
durations = []
corr = {'psb':[], 'lmn':[]}

ccs = {i:[] for i in range(3)}

for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "pynapplenwb", basename + ".nwb")

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

        # hmm_eps = []
        # try:
        #     filepath = os.path.join(data_directory, s, os.path.basename(s))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep0.npz"))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep1.npz"))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep2.npz"))
        # except:
        #     pass
        
        psb_spikes = spikes[spikes.location=="psb"]

        spikes = spikes[(spikes.location=="psb")|(spikes.location=="lmn")]

        if len(spikes):
            ############################################################################################### 
            # COMPUTING TUNING CURVES
            ###############################################################################################
            tuning_curves = nap.compute_1d_tuning_curves(
                spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                ep = position.time_support.loc[[0]])
            tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
            
            SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
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

            # try:
            #     maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
                
            # except:
            #     meanwf, maxch = data.load_mean_waveforms(spike_count=100)
            #     maxch.to_csv(data.nwb_path + "/maxch.csv")        
            
            spikes          = spikes[tokeep]
            tcurves         = tuning_curves[tokeep]
            peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
            spikes['peaks'] = peaks

            # spikes.set_info(maxch = maxch[tokeep])
            if np.sum(spikes.location=="lmn") > 4:
                
                allspikes = spikes
                
                spikes = spikes[spikes.location=='lmn']
                spikes['order'] = np.argsort(spikes.peaks)
                
                # figure()
                # for i in range(len(tokeep)):
                #     subplot(4, 6, i+1, projection='polar')
                #     plot(tcurves[tokeep[i]])
                #     title(allspikes.location.values[i])
                
                try: 
                    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                    newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
                except:
                    velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                    newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)
                
                ############################################################################################### 
                # HMM GLM
                ###############################################################################################
                
                bin_size = 0.010
                window_size = bin_size*50.0
                

                # GLM
                ############################################
                print("fitting GLM")

                basis = nmo.basis.RaisedCosineLogConv(
                    n_basis_funcs=3, window_size=int(window_size/bin_size), conv_kwargs={'shift':False}
                    )

                mask = np.repeat(1-np.eye(len(spikes)), 3, axis=0)

                # Ring
                Y = spikes.count(bin_size, newwake_ep)
                glm = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", feature_mask=mask, solver_name="LBFGS")
                glm.fit(basis.compute_features(Y), Y)

                # Random
                spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(newwake_ep))
                spikes2.set_info(group = spikes._metadata["group"])
                Y2 = spikes2.count(bin_size, newwake_ep)

                rglm = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", feature_mask=mask, solver_name="LBFGS")
                rglm.fit(basis.compute_features(Y2), Y2)
                
                # Null
                spikes0 = nap.TsGroup(
                    {i:nap.Ts(
                        np.sort(np.random.choice(spikes[i].t, int(0.0*len(spikes[i])), replace=False))
                        ) for i in spikes.keys()}, time_support=newwake_ep)
                Y0 = spikes0.count(bin_size, newwake_ep)
                
                glm0 = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", feature_mask=mask, solver_name="LBFGS")
                glm0.fit(basis.compute_features(Y0), Y0)


                # HMM
                ############################################
                
                # hmm = GLM_HMM_nemos((glm0, glm, rglm))
                hmm = GLM_HMM_nemos((glm, rglm))

                # Y = spikes.count(bin_size, sws_ep)
                # X = basis.compute_features(Y)
                
                Yo = overlap_count(spikes, 0.01, 0.008, sws_ep[0:5])
                Xo = basis.compute_features(Yo)
                # sys.exit()

                hmm.fit_transition(Xo, Yo)
                
                ##################################################
                # PSB FIRING AT Transition
                for e in hmm.eps.keys():
                    cc = nap.compute_eventcorrelogram(psb_spikes, nap.Ts(hmm.eps[e].start), 0.01, 1, norm=True)
                    cc.columns = [basename+"_"+str(i) for i in cc.columns]

                    ccs[e].append(cc)


                durations.append(pd.DataFrame(data=[e.tot_length('s') for e in hmm.eps.values()], columns=[basename]).T)

for e in ccs.keys():
    ccs[e] = pd.concat(ccs[e], axis=1)

durations = pd.concat(durations)

figure()
for e in ccs.keys():
    subplot(1,len(ccs),e+1)
    tmp = ccs[e].apply(zscore)
    plot(tmp.mean(1))



figure()
tmp = durations.values
tmp = tmp/tmp.sum(1)[:,None]
plot(tmp.T, '.-', color = 'grey')
plot(tmp.mean(0), 'o-', markersize=20)
ylim(0, 1)
title("Durations")

show()
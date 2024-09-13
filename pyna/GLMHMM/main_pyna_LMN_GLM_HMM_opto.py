# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-08-13 19:04:07
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
# from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
import nemos as nmo
import nwbmatic as ntm
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


datasets = yaml.safe_load(open("/mnt/ceph/users/gviejo/datasets_OPTO.yaml", "r"))

structures = ['lmn', 'adn']

SI_thr = {
    'adn':0.3, 
    'lmn':0.2,
    'psb':1.5
    }


ratio = {}
p_state = {}
count = {}


#%%
for st in structures:    

    ratio[st] = {}
    p_state[st] = {}
    count[st] = {}

    for s in datasets['opto'][f'opto_{st}_psb']['sleep']:
        print(s)

        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        path = os.path.join(data_directory, "OPTO", s)
        if os.path.isdir(os.path.join(path, "pynapplenwb")):

            data = ntm.load_session(path, 'neurosuite')
            spikes = data.spikes
            position = data.position
            wake_ep = data.epochs['wake'].loc[[0]]
            sws_ep = data.read_neuroscope_intervals('sws')
            # rem_ep = data.read_neuroscope_intervals('rem')
            # down_ep = data.read_neuroscope_intervals('down')


            idx = spikes._metadata[spikes._metadata["location"].str.contains(st)].index.values
            spikes = spikes[idx]

            ############################################################################################### 
            # LOADING OPTO INFO
            ###############################################################################################            
            try:
                os.remove(os.path.join(path, os.path.basename(path))+"_opto_ep.npz")
                opto_ep = nap.load_file(os.path.join(path, os.path.basename(path))+"_opto_ep.npz")
            except:
                opto_ep = []
                epoch = 0
                while len(opto_ep) == 0:
                    try:
                        opto_ep = loadOptoEp(path, epoch=epoch, n_channels = 2, channel = 0)
                        opto_ep = opto_ep.intersect(data.epochs[ep])
                    except:
                        pass                    
                    epoch += 1
                    if epoch == 10:
                        sys.exit()
                opto_ep.save(os.path.join(path, os.path.basename(path))+"_opto_ep")

            
            opto_ep = opto_ep.intersect(sws_ep)



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

            count[st][s] = len(spikes)


            print(s, tokeep)
            if len(tokeep) > 5:
                
                # figure()
                # for i in range(len(tokeep)):
                #     subplot(4, 4, i+1, projection='polar')
                #     plot(tcurves[tokeep[i]])
                
                
                velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                newwake_ep = np.abs(velocity).threshold(0.02).time_support.drop_short_intervals(1).merge_close_intervals(1)

                ############################################################################################### 
                # HMM GLM
                ###############################################################################################
                
                bin_size = 0.02
                window_size = bin_size*50.0
                

                # GLM
                ############################################
                print("fitting GLM")

                basis = nmo.basis.RaisedCosineBasisLog(
                    n_basis_funcs=4, shift=False, mode="conv", window_size=int(window_size/bin_size)
                    )

                mask = np.repeat(1-np.eye(len(spikes)), basis.n_basis_funcs, axis=0)

                Y = spikes.count(bin_size, newwake_ep)

                spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(newwake_ep))
                spikes2.set_info(maxch = spikes._metadata["maxch"], group = spikes._metadata["group"])
                Y2 = spikes2.count(bin_size, newwake_ep)

                spikes0 = nap.TsGroup({i:nap.Ts(np.array([])) for i in spikes.keys()}, time_support=newwake_ep)
                Y0 = spikes0.count(bin_size, newwake_ep)

                glm = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"), feature_mask=mask)
                glm.fit(basis.compute_features(Y), Y)

                rglm = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"), feature_mask=mask)
                rglm.fit(basis.compute_features(Y2), Y2)

                glm0 = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"), feature_mask=mask)
                glm0.fit(basis.compute_features(Y0), Y0)

                # HMM
                ############################################
                
                hmm = GLM_HMM_nemos((glm0, glm, rglm))
                # hmm = GLM_HMM((glm, rglm))

                Y = spikes.count(bin_size, sws_ep)
                X = basis.compute_features(Y)
                
                hmm.fit_transition(X, Y)


                ################################################
                # RATIO IN opto/ out opto
                ############################################

                tmp = hmm.Z.restrict(opto_ep)
                opto_in = np.zeros(3)
                for i in range(3):
                    opto_in[i] = np.sum(tmp == i)/len(tmp)
                
                tmp = hmm.Z.restrict(sws_ep.set_diff(opto_ep))
                opto_out = np.zeros(3)
                for i in range(3):
                    opto_out[i] = np.sum(tmp == i)/len(tmp)
                
                ratio[st][s] = (opto_in-opto_out)/(opto_in+opto_out)

                ################################################
                # Perievent
                ################################################

                pevent = nap.compute_perievent_continuous(hmm.Z, nap.Ts(opto_ep.start), (1, 2))

                p_state[st][s] = nap.TsdFrame(
                    t = pevent.t,
                    d = np.array([np.mean(pevent==i, 1).d for i in range(3)]).T
                    )
                

                # break
                # ide = 3
                # ex_ep = nap.IntervalSet(opto_ep[ide,0]-1.0, opto_ep[ide, 1]+1.0)

                # figure()
                # gs = GridSpec(3,1)
                # ax = subplot(gs[0,0])
                # plot(hmm.Z.restrict(ex_ep))       
                # ylabel("state")
                # axvspan(opto_ep[ide,0], opto_ep[ide,1], color='red', alpha=0.1)

                # subplot(gs[1,0], sharex=ax)
                # plot(spikes.restrict(ex_ep).to_tsd("peaks"), '|', markersize=5)
                # ylabel("Spikes")
                # ylim(0, 2*np.pi)
                # axvspan(opto_ep[ide,0], opto_ep[ide,1], color='red', alpha=0.1)

#%%




from scipy.stats import sem





labels = ['0-state', 'attractor', 'random']

colors = ['blue', 'red',  'green']


figure(figsize = (15, 8))
subplot(131)
for j, st in enumerate(ratio.keys()):
    tmp = pd.DataFrame.from_dict(ratio[st]).T
    for i in range(3):        
        plot(np.ones(len(tmp))*i + 0.5*j +np.random.randn(len(tmp))*0.05, tmp.values[:,i], 'o', color = colors[i])
    # plot(np.ones(len(ratio_in))*i+np.random.randn(len(ratio_in))*0.1, ratio_in[:,i], 'o')
    # plot(np.ones(len(ratio_out))*i+np.random.randn(len(ratio_out))*0.1+0.25, ratio_out[:,i], 'o')
xticks(np.arange(3), labels)

for i, st in enumerate(p_state.keys()):

    subplot(1,3,i+2)

    mps = np.mean(np.array([p_state[st][s].values for s in p_state[st].keys()]), 0)
    t = list(p_state[st].values())[0].t
    # sps = sem(pstate, 0)

    for i in range(3):
        # fill_between(t, mps[:,i]-sps[:,i], mps[:,i]+sps[:,i], color = colors[i], alpha=0.1)
        plot(t, mps[:,i], label = labels[i], color=colors[i])
    xlabel("Time (s)")
    ylabel("P state")
    legend()
    title(st)
    ylim(0, 1)



# savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_OPTO_GLMHMM.png"))
show()

# %%

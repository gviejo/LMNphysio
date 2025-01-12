# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   gviejo
# @Last Modified time: 2025-01-07 12:37:47
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
import nemos as nmo
from scipy.stats import poisson

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
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'

# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

datasets = np.hstack([
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])

sta = {'psb':[], 'lmn':[]}


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

        sws_ep = nap.IntervalSet(sws_ep.start, sws_ep.end)
        # sys.exit()

        spikes = spikes[(spikes.location == "psb") | (spikes.location == "lmn")]
        
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
            if np.sum(spikes.location=="lmn") > 3:
                
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
                
                bin_size = 0.005
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


                Y = spikes.count(bin_size, sws_ep)
                X = basis.compute_features(Y)
                             
                mu = glm.predict(X)
                p = poisson.pmf(k=Y, mu=mu)
                p = p - np.nanmean(p, 0)
                p = p / np.nanstd(p, 0)

                P = nap.Tsd(t=X.t, d=np.nanmean(p, 1), time_support=X.time_support)                
                P = P.dropna()

                ###################################################
                # STA PSB
                ###################################################

                sta_psb = nap.compute_event_trigger_average(allspikes[allspikes.location=='psb'], P, 0.01, 1, sws_ep)
                sta_lmn = nap.compute_event_trigger_average(allspikes[allspikes.location=='lmn'], P, 0.01, 1, sws_ep)                


                sta['psb'].append(sta_psb.as_dataframe())
                sta['lmn'].append(sta_lmn.as_dataframe())


for k in sta:
    sta[k] = pd.concat(sta[k], axis=1)


figure()
for i, k in enumerate(sta.keys()):
    subplot(1,2,i+1)
    tmp = sta[k].apply(zscore)
    plot(tmp)
    title(k)
show()
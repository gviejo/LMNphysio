# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-01-04 06:11:33
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-04-07 16:58:23
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore
import nemos as nmo
# import nwbmatic as ntm
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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')


allsi = []
alleta = []


for s in datasets:

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

        
        psb_spikes = spikes[spikes.location=="psb"]

        spikes = spikes[(spikes.location=="psb")|(spikes.location=="lmn")]
        
        
        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
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
        
        spikes = spikes[tokeep]


        psb = spikes.location[spikes.location=="psb"].index.values
        lmn = spikes.location[spikes.location=="lmn"].index.values

    
        print(s)
        
        tcurves         = tuning_curves[tokeep]
        # tcurves = tuning_curves
        


        try:
            velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
        except:
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)


        ############################################################################################### 
        # FITTING NEMOS GLM DURING WAKE. COMPUTING P(ATTRACTOR) DURING SWS
        ###############################################################################################         
        bin_size = 0.02
        window_size = bin_size*50.0


        basis = nmo.basis.RaisedCosineLogConv(
            n_basis_funcs=3, window_size=int(window_size/bin_size), conv_kwargs={'shift':False}
            )

        mask = np.repeat(1-np.eye(len(lmn)), 3, axis=0)
        

        # Ring
        Y = spikes[spikes.location=="lmn"].count(bin_size, newwake_ep)
        Y = np.sqrt(Y/Y.max(0))                
        glm = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", feature_mask=mask, solver_name="LBFGS")
        glm.fit(basis.compute_features(Y), Y)


        Y = spikes[spikes.location=="lmn"].count(bin_size, sws_ep)
        Y = np.sqrt(Y/Y.max(0))
        X = basis.compute_features(Y)

        # Computing P
        mu = glm.predict(X)
        p = poisson.pmf(k=Y, mu=mu)
        p = np.clip(p, 1e-15, 1.0)

        # lp = nap.Tsd(t=X.t, d=np.sum(np.log(p),1))
        lp = nap.Tsd(t=X.t, d=np.log(np.prod(p, 1)))
        lp = lp.dropna()
        lp = lp.smooth(bin_size*3, size_factor=20)
        

        # FIRING RATE POSTSUB
        fr_psb = spikes[(spikes.location=="psb")].to_tsd().count(bin_size, sws_ep)/bin_size
        fr_psb = fr_psb.smooth(bin_size*3, size_factor=20)
        fr_psb = fr_psb.restrict(lp.time_support)

        psb_lp = nap.compute_event_trigger_average(spikes[spikes.location=="psb"], lp, bin_size, 1, sws_ep)
        # psb_lp = nap.compute_perievent_continuous(lp, spikes[spikes.location=="psb"].to_tsd(), (-1, 1))
    
        psb_lp = psb_lp.as_dataframe()
        cols = [s.split("/")[-1] + "_" + str(i) for i in psb_lp.columns]
        psb_lp.columns = cols

        alleta.append(psb_lp)

        allsi.append(pd.Series(index=cols, data=spikes[(spikes.location=="psb")].SI.values))



allsi = pd.concat(allsi)
alleta = pd.concat(alleta, axis=1)

alleta = alleta - alleta.mean(0)
alleta = alleta / alleta.std(0)


figure()
plot(alleta)


show()
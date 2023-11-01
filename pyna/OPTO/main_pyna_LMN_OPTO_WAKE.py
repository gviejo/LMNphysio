# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-29 15:43:45
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-11-01 12:30:22
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
# from sklearn.linear_model import PoissonRegressor



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


datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_OPTO_WAKE.list'), delimiter = '\n', dtype = str, comments = '#')


SI_thr = {
    'adn':0.5, 
    'lmn':0.2,
    'psb':1.5
    }

allfr = []
alltcn = []
alltco = []
allmeta = []


for s in datasets:
# for s in ["A8000/A8047/A8047-230310B"]:
    print(s)    
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sleep_ep = data.epochs["sleep"]


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

    print(s, len(tokeep))

    if len(tokeep):
        ############################################################################################### 
        # LOADING OPTO INFO
        ###############################################################################################    
        try:
            opto_ep = nap.load_file(os.path.join(path, os.path.basename(path))+"_opto_wake_ep.npz")
        except:
            opto_ep = []
            epoch = 0
            while len(opto_ep) == 0:
                try:
                    opto_ep = loadOptoEp(path, epoch=epoch, n_channels = 2, channel = 0)
                    opto_ep = opto_ep.intersect(data.epochs["wake"])
                except:
                    pass                    
                epoch += 1
                if epoch == 10:
                    sys.exit()
            opto_ep.save(os.path.join(path, os.path.basename(path))+"_opto_wake_ep")

        ############################################################################################### 
        # FIRING RATE MODULATION
        ###############################################################################################    
        stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

        # peth = nap.compute_perievent(spikes[tokeep], nap.Ts(opto_ep["start"].values), minmax=(-stim_duration, 2*stim_duration))
        # frates = pd.DataFrame({n:peth[n].count(0.05).sum(1) for n in peth.keys()})
        frates = nap.compute_eventcorrelogram(spikes[tokeep], nap.Ts(opto_ep["start"].values), stim_duration/20., stim_duration*2, norm=True)
        frates.columns = [data.basename+"_"+str(i) for i in frates.columns]

        if int(stim_duration) == 1:
            sys.exit()

        ############################################################################################### 
        # TUNING CURVES modulation
        ###############################################################################################     
        wake2_ep = wake_ep.set_diff(opto_ep.merge_close_intervals(11.0))

        tc_opto = nap.compute_1d_tuning_curves(spikes[tokeep], position['ry'], 120, minmax=(0, 2*np.pi), ep = opto_ep)
        tc_opto = smoothAngularTuningCurves(tc_opto, window = 20, deviation = 2.0)
        SI_opto = nap.compute_1d_mutual_info(tc_opto, position['ry'], opto_ep, (0, 2*np.pi))
        tc_opto.columns = [data.basename+"_"+str(i) for i in tc_opto.columns]
        SI_opto.index = tc_opto.columns
        

        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = wake2_ep)
        tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)
        SI2 = nap.compute_1d_mutual_info(tuning_curves, position['ry'], wake2_ep, (0, 2*np.pi))
        tuning_curves.columns = [data.basename+"_"+str(i) for i in tuning_curves.columns]
        SI2.index = tuning_curves.columns
        # if len(tokeep) > 2:

        velocity = computeAngularVelocity(position['ry'], wake_ep, 0.1)
        atiopto = computeLMN_TC(spikes, position['ry'], opto_ep, velocity)
        atitc = computeLMN_TC(spikes, position['ry'], wake2_ep, velocity)

        ctc = {}

        for k in range(3):
            ctc[k] = {}
            for ep, ati in zip(['wake', 'opto'], [atitc, atiopto]):
                new_tcurve = []
                for n in ati.keys():
                    y = ati[n][k]
                    x = y.index.values - y.index[y.index.get_indexer([peaks[n]], method='nearest')[0]]
                    x[x<-np.pi] += 2*np.pi
                    x[x>np.pi] -= 2*np.pi
                    tmp = pd.Series(index = x, 
                        data = y.values/y.values.max()).sort_index()
                    
                    new_tcurve.append(tmp.values)                
                new_tcurve = pd.DataFrame(
                    index = np.linspace(-np.pi, np.pi, y.shape[0]+1)[0:-1], 
                    data = np.array(new_tcurve).T, columns = ati.keys())
                ctc[k][ep] = new_tcurve


        figure()
        gs = GridSpec(3,3)
        for i,n in enumerate(atitc.keys()):        
            gs2 = GridSpecFromSubplotSpec(3, 1, gs[i//3, i%3])
            for k in range(3):
                subplot(gs2[k,:])
                plot(atitc[n].iloc[:,k])
                plot(atiopto[n].iloc[:,k], '--')


        figure()
        for k in range(3):
            subplot(1,3,k+1)
            plot(ctc[k]['wake'].mean(1), color = 'black')
            plot(ctc[k]['opto'].mean(1), '--', color = 'red')

        show()


        sys.exit()
        #######################
        # SAVING
        #######################
        allfr.append(frates)        
        alltcn.append(tuning_curves)        
        alltco.append(tc_opto)
        metadata = pd.DataFrame.from_dict(
            {"SI":SI2["SI"],
            "SIO":SI_opto["SI"]}
            )

        allmeta.append(metadata)
        tcurves.columns = frates.columns        
        

allfr = pd.concat(allfr, 1)
alltcn = pd.concat(alltcn ,1)
alltco = pd.concat(alltco ,1)
allmeta = pd.concat(allmeta, 0)


figure()
gs = GridSpec(1, 2)

subplot(gs[0,0])
plot(np.arange(2), allmeta.values.T, 'o-')
xticks([0, 1], ['wake', 'opto'])

subplot(gs[0,1])
plot(allfr, alpha = 0.2, color = 'grey')
plot(allfr.mean(1), alpha = 1.0, color = 'red')
show()


##################################################################
# FOR FIGURE 1
##################################################################
datatosave = {
    "allfr":allfr,
    "allmeta":allmeta,
    "alltcn":alltcn,
    "alltco":alltco
}


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
file_name = "OPTO_LMN_wake.pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)
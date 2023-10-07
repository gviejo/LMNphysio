# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-11-29 14:59:45
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-09 17:43:11
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
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


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




allr = []

# for s in datasets:
for s in ['LMN-PSB/A3019/A3019-220701A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes.getby_threshold('rate', 0.75)
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    up_ep = read_neuroscope_intervals(data.path, data.basename, 'up')
    down_ep = read_neuroscope_intervals(data.path, data.basename, 'down')


    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.002).time_support         

    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|psb")].index.values
    spikes = spikes[idx]
        
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tcurves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tcurves = smoothAngularTuningCurves(tcurves, 20, 4)
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    spikes.set_info(SI, peaks=peaks)

    psb = list(spikes.getby_category("location")["psb"].getby_threshold("SI", 0.06).index)
    lmn = list(spikes.getby_category("location")["lmn"].getby_threshold("SI", 0.1).index)
    nhd = list(spikes.getby_category("location")["psb"].getby_threshold("SI", 0.04, op='<').index)



    # figure()
    # for i, n in enumerate(spikes.index):
    #     subplot(10,10,i+1, projection='polar')        
    #     if n in psb:
    #         plot(tcurves[n], color = 'red')
    #     elif n in lmn:
    #         plot(tcurves[n], color = 'green')
    #     elif n in nhd:
    #         plot(tcurves[n], color = 'blue')
    #     else:
    #         plot(tcurves[n], color = 'grey')
    #     xticks([])
    #     yticks([])

    # sys.exit()

    


    ############################################################################################### 
    # REACTIVATION
    ###############################################################################################
    R = {}
    for gr, grp in zip(['psb', 'lmn', 'nhd'], [psb, lmn, nhd]):
        count = spikes[grp].count(0.3, wake_ep)
        zref = StandardScaler().fit_transform(count)
        ev, comp = np.linalg.eig((1/zref.shape[0])*np.dot(zref.T, zref))    
        thr = (1 + np.sqrt(zref.shape[1]/zref.shape[0])) ** 2.0
        comp = comp[:,ev>thr]
        # x = np.dot(zref, comp)
        # w = FastICA().fit(x).components_[:,0:3]
        # count = spikes[grp].count(0.02, sws_ep)
        # ztar = StandardScaler().fit_transform(count)

        ztar = zref

        P = []
        for i in range(comp.shape[1]):
            p = comp[:,i][:,np.newaxis]*comp[:,i]
            p[np.diag_indices_from(p)] = 0.0
            P.append(p)
        P = np.array(P)

        r = np.zeros((ztar.shape[0], comp.shape[1]))
        for i in range(len(r)):
            tmp = np.tile((ztar[i][:, np.newaxis] * ztar[i])[np.newaxis,:,:], (comp.shape[1],1,1))
            r[i] = (P * tmp).sum((1,2))    

        r = nap.TsdFrame(t=count.index.values, d = r)

        R[gr] = r

        break

    for gr in R.keys():    
        tc = nap.compute_1d_tuning_curves_continous(R[gr], position["ry"], 60, position["ry"].time_support.loc[[0]], (0, 2 * np.pi))
        tc = smoothAngularTuningCurves(tc, 10, 1)

        figure()
        for i in tc.columns:
            subplot(5,5,i+1, projection='polar')
            plot(tc[i])
            title(gr)
    show()

    

    up_grp = nap.TsGroup({0:nap.Ts(t=up_ep['start'].index.values, time_support=sws_ep)})

    sta = [nap.compute_event_trigger_average(up_grp, r[i], 0.04, (-0.5,0.5), ep=sws_ep)[0] for i in r.columns]
    sta = pd.concat(sta, 1)

    sys.exit()

    figure()
    ax = subplot(411)
    [plot(spikes[n].restrict(sws_ep).fillna(i), '|') for i,n in enumerate(nhd)]    
    subplot(412, sharex=ax)
    [plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|') for i,n in enumerate(psb)]
    # plot(position['ry'].restrict(wake_ep.loc[[0]]))
    subplot(413, sharex=ax)
    [plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|') for i,n in enumerate(lmn)]
    # plot(position['ry'].restrict(wake_ep.loc[[0]]))
    subplot(414, sharex=ax)
    plot(r.loc[:,0])
    show()

    sys.exit()
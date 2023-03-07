# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-01-23 17:40:32
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-24 12:18:18

import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations, product
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from scipy.linalg import hankel
from sklearn.linear_model import PoissonRegressor
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat

def offset_matrix(rate, binsize=0.01, windowsize = 0.1):
    idx1 = -np.arange(0, windowsize + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, windowsize + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))

    # Build the Hankel matrix
    tmp = rate
    n_p = len(idx1)
    n_f = len(idx2)
    pad_tmp = np.pad(tmp, (n_p, n_f))
    offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]        

    return offset_tmp, time_idx

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_PSB.list'), delimiter = '\n', dtype = str, comments = '#')

datasets = [str(datasets)]


coefs_mua = {e:[] for e in ['wak', 'sws', 'rem']}
coefs_pai = {e:[] for e in ['wak', 'sws', 'rem']}

pairs_info = pd.DataFrame(columns = ['offset', 'session'])

alltc = []
allsi = []


for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    wake_ep = data.epochs['wake'].loc[[0]]
    sleep_ep = data.epochs['sleep']
    
    tmp = loadmat(path+'/Analysis/Angle.mat', simplify_cells=True)
    position = nap.TsdFrame(
        t = tmp['ang']['t'],
        d = tmp['ang']['data'],
        columns = ['ry'],
        time_support = wake_ep
        )    
    position = (position + (2 * np.pi)) % (2 * np.pi)
    position = position.fillna(method="ffill")

    angvel = computeAngularVelocity(position['ry'], wake_ep, 0.1)    
    angvel2 = angvel.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=10)
    angvel2 = nap.Tsd(angvel2)
    wake_ep = angvel2.threshold(0.25, 'above').time_support  

    tmp = loadmat(path+'/Analysis/SleepStateEpisodes.states.mat', simplify_cells=True)

    sws_ep = tmp["SleepStateEpisodes"]["ints"]["NREMepisode"]
    sws_ep = nap.IntervalSet(start=sws_ep[:,0], end=sws_ep[:,1])
    sws_ep = sleep_ep.intersect(sws_ep)

    # rem_ep = tmp["SleepStateEpisodes"]["ints"]["REMepisode"]
    # rem_ep = nap.IntervalSet(start=rem_ep[:,0], end=rem_ep[:,1])
    # rem_ep = sleep_ep.intersect(rem_ep)

    tmp = loadmat(path+'/Analysis/CellTypes.mat', simplify_cells=True)

    idx = spikes.index[tmp['hd']==1]
    spikes = spikes[idx]
      
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tcurves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tcurves = smoothAngularTuningCurves(tcurves)
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    spikes.set_info(SI, peaks=peaks)
    
    colnames = [data.basename+'_'+str(i) for i in tcurves.columns]
    tuning_curves = tcurves.copy()
    tuning_curves.columns = colnames
    SI.index = colnames
    
    alltc.append(tuning_curves)
    allsi.append(SI)
    
    # tokeep = list(spikes.getby_category("location")['psb'].getby_threshold("SI", 0.5).getby_threshold("rate", 1).index)
    SI_thr = 1.0
    tokeep = spikes.getby_threshold("SI", SI_thr).index

    # if s == 'LMN-ADN/A5011/A5011-201014A':
    #     sys.exit(0)

    # figure()
    # for i, n in enumerate(tokeep):
    #     subplot(10,10,i+1, projection='polar')        
    #     plot(tcurves[n], color = 'red')            
    #     xticks([])
    #     yticks([])        

    ############################################################################################### 
    # GLM
    ###############################################################################################

    for e, ep, binsize, windowsize in zip(
            ['wak', 'sws'], 
            [wake_ep, sws_ep], 
            [0.02, 0.001],
            [10, 0.1]):
        cc = nap.compute_crosscorrelogram(spikes, binsize, windowsize, ep, norm=True)

        pair_name = [data.basename + '_' + str(p[0]) + '_' + str(p[1]) for p in cc.columns]
        cc.columns = pair_name
        coefs_pai[e].append(cc)

    pairs = list(combinations(tokeep, 2))

    for i, p in enumerate(pairs):
        tar_neuron = p[0]
        reg_neuron = p[1]
        
        pair_name = data.basename + '_' + str(p[0]) + '_' + str(p[1])

        a = peaks[tar_neuron] - peaks[reg_neuron]
        pair_offset = np.abs(np.arctan2(np.sin(a), np.cos(a)))        
        pairs_info.loc[pair_name, 'offset'] = pair_offset
        pairs_info.loc[pair_name, 'session'] = s





alltc = pd.concat(alltc, 1)
allsi = pd.concat(allsi)

allsi = allsi.sort_values(by="SI", ascending=False)
alltc = alltc[allsi.index.values]
alltc = centerTuningCurves(alltc)
alltc = alltc/alltc.max()


figure()
subplot(121)
plot(allsi.values, np.arange(len(allsi)))
xlabel("Spatial Information")
axvline(SI_thr)
subplot(122)
imshow(alltc.values.T, aspect='auto', origin = 'lower')
xlabel("Centered\ntuning curves")
# show()






pairs_info = pairs_info.sort_values(by="offset")
for k in ['wak', 'sws']:    
    coefs_pai[k] = pd.concat(coefs_pai[k], 1)            
    coefs_pai[k] = coefs_pai[k][pairs_info.index]



figure()
gs = GridSpec(2,2)


inters = np.linspace(0, np.pi, 6)
idx = np.digitize(pairs_info['offset'], inters)-1

for j, k in enumerate(['wak', 'sws']):
    subplot(gs[0,j])
    tmp = coefs_pai[k].rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)    
    #tmp = tmp.apply(zscore)
    tmp = gaussian_filter(tmp.values.T, (1,1))
    imshow(tmp, aspect='auto', cmap = 'jet')
    title(k)

    subplot(gs[1,j])
    for l in range(len(inters)-1):
        tmp = coefs_pai[k].iloc[:,idx==l].rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        #tmp = tmp.apply(zscore)        
        plot(tmp.mean(1), '-')            



celldep = {}
for s in datasets:
    celldepth = scipy.io.loadmat("/mnt/DataRAID2/"+s+"/Analysis/CellDepth.mat")    
    celldep[s.split("/")[-1]] = celldepth['cellDep'].flatten()

pairs_info['distance'] = np.nan
for p in pairs_info.index:
    s = p.split("_")[0]
    d = celldep[s][int(p.split("_")[1])] - celldep[s][int(p.split("_")[2])]
    pairs_info.loc[p,'distance'] = d


order = pairs_info[pairs_info["offset"] < (np.pi/6)].sort_values(by="distance").index.values
#order = pairs_info.sort_values(by="distance").index.values

figure()
gs = GridSpec(2,2)

cut = {'wak':1, 'sws':0.05}

for j, k in enumerate(['wak', 'sws']):
    subplot(gs[0,j])

    betalag = (coefs_pai[k][order]
        .rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0)
        .mean(std=20).loc[-cut[k]:cut[k]].idxmax())

    scatter(betalag.values, pairs_info.loc[order, "distance"].values)

    m, b = np.polyfit(pairs_info.loc[order, "distance"].values, betalag.values, 1)
    tmp = pairs_info.loc[order, "distance"].values
    y = np.linspace(tmp.min(), tmp.max(), 5)
    plot(y*m + b, y)

    xlabel("Beta peak\nTime lag")
    ylabel("Relativee distance\n ")

    title(k)

    subplot(gs[1,j])

    # betalag = coefs_pai[k][order].apply(zscore).loc[-cut[k]:cut[k]].values.T
    betalag = coefs_pai[k][order].loc[-cut[k]:cut[k]].values
    betalag = betalag/betalag.max(0)    

    # tmp = gaussian_filter(betalag.T, (1,1))
    tmp = betalag.T

    imshow(tmp, cmap='jet', aspect='auto', origin='lower')

    xlabel("Beta peak\nTime lag")
    ylabel("Relative distance\n ")

    title(k)


show()

#!/usr/bin/env python
'''

'''
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



############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')



allcc = {'wak':[], 'rem':[], 'sws':[]}
allccdown = {'psb':[], 'lmn':[]}

for s in datasets:
#for s in ['LMN-psb/A5011/A5011-201015A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    down_ep = data.read_neuroscope_intervals('down')
    idx = spikes._metadata[spikes._metadata["location"].str.contains("psb|lmn")].index.values
    spikes = spikes[idx]
    
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves)    
    tcurves = tuning_curves
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    spikes.set_info(SI, peaks=peaks)

    psb = list(spikes.getby_category("location")["psb"].getby_threshold("SI", 0.5).index)
    lmn = list(spikes.getby_category("location")["lmn"].getby_threshold("SI", 0.1).index)
    # nhd = list(spikes.getby_category("location")["psb"].getby_threshold("SI", 0.04, op='<').index)

    
    tokeep = psb+lmn
    tokeep = np.array(tokeep)
    spikes = spikes[tokeep]
    groups = spikes._metadata.loc[tokeep].groupby("location").groups

    # tcurves       = tuning_curves[tokeep]
    # peaks             = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.001).time_support 

    ############################################################################################### 
    # MUA CROSS CORRELOGRAM
    ############################################################################################### 
    groups = spikes.getby_category("location")

    if len(groups['psb'])>4 and len(groups['lmn'])>4:
        mua = {}
        for i, n in enumerate(['lmn', 'psb']):
            mua[i] = nap.Ts(t=np.sort(np.hstack([groups[n][j].index.values for j in groups[n].index])))
        mua = nap.TsGroup(mua, time_support = spikes.time_support)
        
        for e, ep, bin_size, window_size in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], [0.01, 0.01, 0.01], [1, 1, 1]):
            allcc[e].append(nap.compute_crosscorrelogram(mua, bin_size, window_size, ep, norm=True)[(0,1)])

        ######################################################################################################
        # Cross-correlation DOWN center
        ######################################################################################################    
        # TAKING UP_EP AND DOWN_EP LARGER THAN 100 ms
        down_ep = down_ep.drop_short_intervals(50, time_units = 'ms')  

        tref = nap.Ts(
            t=down_ep['start'].values + (down_ep['end'].values - down_ep['start'].values)/2
            )

        for k in ['psb', 'lmn']:
            cc_down = nap.compute_eventcorrelogram(groups[k], tref, 0.01, 0.2, sws_ep)

            cc_down.columns = [data.basename+'_'+str(n) for n in groups[k].keys()]

            allccdown[k].append(cc_down)


for k in ['psb', 'lmn']:
    allccdown[k] = pd.concat(allccdown[k], axis=1)

for e in allcc.keys():
    allcc[e] = pd.concat(allcc[e], 1)
    allcc[e] = allcc[e].apply(zscore)

datatosave = {'allcc':allcc, 'allcup': allccdown}
cPickle.dump(datatosave, open(os.path.join('/home/guillaume/Dropbox/CosyneData', 'MUA_LMN_psb.pickle'), 'wb'))


figure()
for i, e in enumerate(allcc.keys()):
    subplot(1,3,i+1)
    #plot(allcc[e], alpha = 0.7, color = 'grey')
    plot(allcc[e].mean(1))
show()

figure()
for i,k in enumerate(['psb', 'lmn']):
    subplot(2,1,i+1)
    plot(allccdown[k].mean(1))
show()

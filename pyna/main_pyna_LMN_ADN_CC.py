# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-01-06 17:03:41
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-01-06 19:58:32
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




############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

# infos = getAllInfos(data_directory, datasets)

allcc_wak = []
allcc_rem = []
allcc_sws = []
allpairs = []
alltcurves = []
allfrates = []
allvcurves = []
allscurves = []
allpeaks = []

for s in datasets:
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
    idx = spikes._metadata[spikes._metadata["location"].str.contains("adn|lmn")].index.values
    spikes = spikes[idx]
    
    name = data.basename

    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves)
    
    # CHECKING HALF EPOCHS
    wake2_ep = splitWake(position.time_support.loc[[0]])    
    tokeep2 = []
    stats2 = []
    tcurves2 = []   
    for i in range(2):
        tcurves_half = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tcurves_half = smoothAngularTuningCurves(tcurves_half)

        tokeep, stat = findHDCells(tcurves_half)
        tokeep2.append(tokeep)
        stats2.append(stat)
        tcurves2.append(tcurves_half)       
    tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
    

    spikes = spikes[tokeep]
    groups = spikes._metadata.loc[tokeep].groupby("location").groups

    tcurves         = tuning_curves[tokeep]
    peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.001).time_support 
        
    ############################################################################################### 
    # CROSS CORRELATION
    ###############################################################################################
    cc_wak = nap.compute_crosscorrelogram(spikes, 0.1,  3, newwake_ep, reverse=True)
    cc_rem = nap.compute_crosscorrelogram(spikes, 0.1,  3, rem_ep, reverse=True)    
    cc_sws = nap.compute_crosscorrelogram(spikes, 0.01,  0.2,  sws_ep, reverse=True)

    cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 1.0)
    cc_rem = cc_rem.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 1.0)
    cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 1.0)

    tcurves                             = tuning_curves[tokeep]
    peaks                               = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()      
    tcurves                             = tcurves[peaks.index.values]
    neurons                             = [name+'_'+str(n) for n in tcurves.columns.values]
    peaks.index                         = pd.Index(neurons)
    tcurves.columns                     = pd.Index(neurons)

    new_index = [(name+'_'+str(i),name+'_'+str(j)) for i,j in cc_wak.columns]
    cc_wak.columns = pd.Index(new_index)
    cc_rem.columns = pd.Index(new_index)
    cc_sws.columns = pd.Index(new_index)

    angdiff = np.zeros(len(new_index))
    for k, (i,j) in enumerate(new_index):
        a = peaks[i] - peaks[j]
        angdiff[k] = np.abs(np.arctan2(np.sin(a), np.cos(a)))        

    pairs = pd.DataFrame(
        index = pd.MultiIndex.from_tuples(new_index), 
        columns = ['struct']
        )
    pairs['ang diff'] = angdiff    

    adn = groups['adn'].astype("str")
    lmn = groups['lmn'].astype("str")

    for p in pairs.index:
        if p[0].split("_")[1] in adn and p[1].split("_")[1] in adn:            
            pairs.loc[p,'struct'] = 'adn-adn'
        elif p[0].split("_")[1] in lmn and p[1].split("_")[1] in lmn:
            pairs.loc[p,'struct'] = 'lmn-lmn'
        elif p[0].split("_")[1] in lmn and p[1].split("_")[1] in adn:
            pairs.loc[p,'struct'] = 'lmn-adn'


    #######################
    # SAVING
    #######################
    alltcurves.append(tcurves)
    allpairs.append(pairs)
    allcc_wak.append(cc_wak[pairs.index])
    allcc_rem.append(cc_rem[pairs.index])
    allcc_sws.append(cc_sws[pairs.index])
    allpeaks.append(peaks)

 
alltcurves  = pd.concat(alltcurves, 1)
allpairs    = pd.concat(allpairs, 0)
allcc_wak   = pd.concat(allcc_wak, 1)
allcc_rem   = pd.concat(allcc_rem, 1)
allcc_sws   = pd.concat(allcc_sws, 1)
allpeaks    = pd.concat(allpeaks, 0)



sess_groups = pd.DataFrame(pd.Series({k:k.split("_")[0] for k in alltcurves.columns.values})).groupby(0).groups


colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(sess_groups)))

datatosave = {  'tcurves':alltcurves,
                'sess_groups':sess_groups,
                'cc_wak':allcc_wak,
                'cc_rem':allcc_rem,
                'cc_sws':allcc_sws,
                'pairs':allpairs,
                'peaks':allpeaks
                }

# cPickle.dump(datatosave, open(os.path.join('../data', 'All_crosscor_ADN_LMN.pickle'), 'wb'))



from matplotlib import gridspec

##########################################################
# TUNING CURVES
figure()
count = 1
for i, g in enumerate(sess_groups.keys()):
    for j, n in enumerate(sess_groups[g]):
        subplot(13,20,count, projection = 'polar')
        plot(alltcurves[n], color = colors[i])
        # title(n)
        xticks([])
        yticks([])
        count += 1

##########################################################
# CROSS CORR
titles = ['wake', 'REM', 'NREM']
figure()
gs = gridspec.GridSpec(3, 5)

for i, st in enumerate(['adn-adn', 'lmn-adn', 'lmn-lmn']):
    subpairs = allpairs[allpairs['struct']==st]
    group = subpairs.sort_values(by='ang diff').index.values
    subplot(gs[i,0])
    plot(allpairs.loc[group, 'ang diff'].values, np.arange(len(group))[::-1])
    ylabel(st)
    for j, cc in enumerate([allcc_wak, allcc_rem, allcc_sws]):
        subplot(gs[i,j+1])
        tmp = cc[group]     
        # tmp = tmp - tmp.mean(0)
        # tmp = tmp / tmp.std(0)
        tmp = scipy.ndimage.gaussian_filter(tmp.T, (1,1))

        imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
        
        title(titles[j])
        xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])

    gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec = gs[i,-1])    
    idx = np.digitize(allpairs.loc[group]['ang diff'], np.linspace(0, np.pi, 4))-1
    subgroup = allpairs.loc[group].groupby(idx).groups
    for j in range(3):
        subplot(gs2[j,0])
        plot(allcc_sws[subgroup[j]].mean(1))

show()



# figure()
# subplot(121)
# cc = allcc_wak[group]
# cc = cc - cc.mean(0)
# cc = cc / cc.std(0)
# cc = cc.loc[-50:50]
# tmp = scipy.ndimage.gaussian_filter(cc.T.values, (1, 1))
# imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')

# subplot(122)
# cc = allcc_sws[group]
# cc = cc - cc.mean(0)
# cc = cc / cc.std(0)
# cc = cc.loc[-50:50]
# tmp = scipy.ndimage.gaussian_filter(cc.T.values, (1, 1))
# imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
# show()




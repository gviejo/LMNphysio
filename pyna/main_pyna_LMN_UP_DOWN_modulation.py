# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 16:12:01
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-07-07 16:48:46
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



############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/Data2/'
datasets = np.genfromtxt('/mnt/DataGuillaume/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)



allmua = []

allccup = []

psb = []
lmn = []

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
    up_ep = data.read_neuroscope_intervals('up')
    down_ep = data.read_neuroscope_intervals('down')    

    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn|psb")].index.values
    spikes = spikes[idx]
    
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
    
    # CHECKING HALF EPOCHS
    wake2_ep = splitWake(position.time_support.loc[[0]])    
    tokeep2 = []
    stats2 = []
    tcurves2 = []   
    for i in range(2):
        tcurves_half = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

        tokeep, stat = findHDCells(tcurves_half)
        tokeep2.append(tokeep)
        stats2.append(stat)
        tcurves2.append(tcurves_half)       
    tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
    
    spikes = spikes[tokeep]
    groups = spikes._metadata.loc[tokeep].groupby("location").groups
    
    psb += [data.basename+'_'+str(n) for n in groups['psb']]
    lmn += [data.basename+'_'+str(n) for n in groups['lmn']]

    tcurves         = tuning_curves[tokeep]
    peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.005).time_support

    # TAKING UP_EP AND DOWN_EP LARGER THAN 100 ms
    up_ep = up_ep.drop_short_intervals(100, time_units = 'ms')
    down_ep = down_ep.drop_short_intervals(100, time_units = 'ms')  

    ######################################################################################################
    # Cross-correlation Up start
    ######################################################################################################    
    up_start = nap.Ts(up_ep["start"].values)
    cc_up = nap.compute_eventcorrelogram(spikes, up_start, 0.01, 0.2, sws_ep)

    cc_up.columns = [data.basename+'_'+str(n) for n in spikes.keys()]

    allccup.append(cc_up)

allccup = pd.concat(allccup, axis=1)

figure()
subplot(221)
plot(allccup[psb], alpha=0.5, color='grey')
title("PSB")
xlabel("Time from Up onset")
subplot(222)
plot(allccup[lmn], alpha=0.5, color='green')
title("LMN")
xlabel("Time from Up onset")
subplot(223)
plot(allccup[psb].mean(1), label = 'psb', color='grey')
plot(allccup[lmn].mean(1), label = 'lmn', color = 'green')
legend()
xlabel("Time from Up onset")
show()


#     ######################################################################################################
#     # HD RATES / UP DOWN    
#     ######################################################################################################    
#     mua = []
#     bins = np.hstack((np.linspace(0,1,200)-1,np.linspace(0,1,200)[1:])) 
#     for n in spikes.keys():
#         spk = spikes[n].restrict(up_ep).index.values
#         spk2 = np.array_split(spk, 10)

#         start_to_spk = []
#         for i in range(len(spk2)):
#             tmp1 = np.vstack(spk2[i]) - up_ep['start'].values
#             tmp1 = tmp1.astype(np.float32).T
#             tmp1[tmp1<0] = np.nan
#             start_to_spk.append(np.nanmin(tmp1, 0))
#         start_to_spk = np.hstack(start_to_spk)

#         spk_to_end = []
#         for i in range(len(spk2)):
#             tmp2 = np.vstack(up_ep['end'].values) - spk2[i]
#             tmp2 = tmp2.astype(np.float32)
#             tmp2[tmp2<0] = np.nan
#             spk_to_end.append(np.nanmin(tmp2, 0))
#         spk_to_end = np.hstack(spk_to_end)

#         d = start_to_spk/(start_to_spk+spk_to_end)
#         mua_up = d

#         spk = spikes[n].restrict(down_ep).index.values
#         tmp1 = np.vstack(spk) - down_ep['start'].values
#         tmp1 = tmp1.astype(np.float32).T
#         tmp1[tmp1<0] = np.nan
#         start_to_spk = np.nanmin(tmp1, 0)

#         tmp2 = np.vstack(down_ep['end'].values) - spk
#         tmp2 = tmp2.astype(np.float32)
#         tmp2[tmp2<0] = np.nan
#         spk_to_end = np.nanmin(tmp2, 0)

#         d = start_to_spk/(start_to_spk+spk_to_end)
#         mua_down = d

#         p, _ = np.histogram(np.hstack((mua_down-1,mua_up)), bins)

#         mua.append(p)

#     mua = pd.DataFrame(
#         index = bins[0:-1]+np.diff(bins)/2, 
#         data = np.array(mua).T,
#         columns = [data.basename+'_'+str(n) for n in spikes.keys()])

#     allmua.append(mua)

# allmua = pd.concat(allmua, 1)
# allmua = allmua/allmua.sum(0)

# figure()
# plot(allmua, alpha = 0.5, color = 'grey')
# plot(allmua.mean(1))

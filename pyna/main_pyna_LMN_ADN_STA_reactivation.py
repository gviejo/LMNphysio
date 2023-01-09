# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-12-22 17:19:39
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-23 15:13:59
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
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

infos = getAllInfos(data_directory, datasets)

sta_r = {'adn':[], 'lmn':[]} # trigger average of reactivtion from the other structure spikes
cc_down = {'adn':[], 'lmn':[]} # cross corr of adn and lmn / adn down states
sta_r_down = {'adn':[], 'lmn':[]} # reactivation trigger on down states



for s in datasets:
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
    idx = spikes._metadata[spikes._metadata["location"].str.contains("adn|lmn")].index.values
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

    adn = list(spikes.getby_category("location")["adn"].getby_threshold("SI", 0.1).index)
    lmn = list(spikes.getby_category("location")["lmn"].getby_threshold("SI", 0.1).index)

    # figure()
    # for i, n in enumerate(spikes.index):
    #     subplot(10,10,i+1, projection='polar')        
    #     if n in adn:
    #         plot(tcurves[n], color = 'red')
    #     elif n in lmn:
    #         plot(tcurves[n], color = 'green')
    #     else:
    #         plot(tcurves[n], color = 'grey')
    #     xticks([])
    #     yticks([])

    # sys.exit()

    tokeep = adn+lmn
    tokeep = np.array(tokeep)
    spikes = spikes[tokeep]    

    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
    newwake_ep = velocity.threshold(0.001).time_support 

    ############################################################################################### 
    # REACTOvATOPM
    ############################################################################################### 
    groups = spikes.getby_category("location")

    if len(groups['adn'])>5 and len(groups['lmn'])>5:

        ## MUA ########
        mua = {
            0:nap.Ts(t=np.sort(np.hstack([groups['lmn'][j].index.values for j in groups['lmn'].index]))),
            1:nap.Ts(t=np.sort(np.hstack([groups['adn'][j].index.values for j in groups['adn'].index])))}

        mua = nap.TsGroup(mua, time_support = spikes.time_support)

        ## DOWN CENTER ######
        down_center = (down_ep["start"] + (down_ep['end'] - down_ep['start'])/2).values
        down_center = nap.TsGroup({
            0:nap.Ts(t=down_center, time_support = sws_ep)
            })

        ## REACTIVATION ####
        bin_size_wake = 0.2
        bin_size_sws = 0.02

        gmap = {'adn':'lmn', 'lmn':'adn'}

        for i, g in enumerate(gmap.keys()):
            #  WAKE 
            count = groups[g].count(bin_size_wake, newwake_ep)
            rate = count/bin_size_wake
            rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            rate = zscore_rate(rate)
            C = (1/rate.shape[0])*np.dot(rate.values.T, rate.values)
            #C[np.diag_indices_from(C)] = 0.0
            ev, comp = np.linalg.eig(C)
            thr = (1 + np.sqrt(rate.shape[1]/rate.shape[0])) ** 2.0    
            comp = comp[:,ev>thr]            

            # SWS
            count = groups[g].count(bin_size_sws, sws_ep)
            rate = count/bin_size_sws
            rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            rate = zscore_rate(rate)            

            # p = np.sum(np.dot(rate.values, C) * rate.values, 1)
            # p = nap.Tsd(t=count.index.values, d = p, time_support = sws_ep)
            p = []
            for j in range(comp.shape[1]):
            #for j in range(1):
                C = comp[:,j][:,np.newaxis]*comp[:,j]
                C[np.diag_indices_from(C)] = 0.0
                p.append(np.sum(np.dot(rate.values, C) * rate.values, 1))
            p = nap.Tsd(t=count.index.values, d = np.array(p).sum(0), time_support = sws_ep)                
        
            # STA / neurons        
            sta_neurons = nap.compute_event_trigger_average(groups[gmap[g]], p, bin_size_sws, (-0.4, 0.4), sws_ep)
            # sta = nap.compute_event_trigger_average(mua[[0]], p, 0.02, (-0.4, 0.4), sws_ep)        
            sta_neurons = sta_neurons.as_dataframe().apply(zscore)        
            
            # STA / down
            sta_down = nap.compute_event_trigger_average(down_center, p, bin_size_sws, (-0.4, 0.4), sws_ep)
            sta_down = sta_down.as_dataframe().apply(zscore)        

            # CC / down
            cc_d = nap.compute_eventcorrelogram(groups[g], down_center[0], bin_size_sws, 0.4, ep=sws_ep)

            ### SAVING ####
            sta_r[g].append(sta_neurons) # trigger average of reactivtion from the other structure spikes
            cc_down[g].append(cc_d) # cross corr of adn and lmn / adn down states
            sta_r_down[g].append(sta_down) # reactivation trigger on down states


for i, g in enumerate(['adn', 'lmn']):
    sta_r[g] = pd.concat(sta_r[g], 1)
    sta_r_down[g] = pd.concat(sta_r_down[g], 1)
    cc_down[g] = pd.concat(cc_down[g], 1)

# sys.exit()


figure()
subplot(1,3,1)
plot(sta_r['adn'].mean(1), label = 'adn r')
plot(sta_r['lmn'].mean(1), label = 'lmn r')
legend()
xlabel("Time from the other")
title("STA reactivation")

subplot(1,3,2)
plot(sta_r_down['adn'].mean(1), label = 'adn r')
plot(sta_r_down['lmn'].mean(1), label = 'lmn r')
xlabel("Time from ADN down center")
title("STA reactivation")
legend()

subplot(1,3,3)
plot(cc_down['adn'].mean(1), label = 'adn')
plot(cc_down['lmn'].mean(1), label = 'lmn')
legend()
xlabel("Time from ADN down center")
title("CC")
show()


sys.exit()

subplot(2,)

subplot(221)
plot(cc1.mean(1))
xlabel("LMN/ADN")
subplot(223)
plot(stas.mean(1))
m = stas.mean(1).values
s = stas.sem(1).values
# s = stas.std(1).values
x = stas.index.values
fill_between(x, m-s, m+s, alpha=0.3)
title("STA ADN reactivation")
ylabel("z")
xlabel("Time lag from LMN")

subplot(222)
plot(cc2.mean(1))
xlabel("ADN/LMN")
subplot(224)
plot(stas2.mean(1))
m = stas2.mean(1).values
s = stas2.sem(1).values
# s = stas.std(1).values
x = stas2.index.values
fill_between(x, m-s, m+s, alpha=0.3)
title("STA LMN reactivation")
ylabel("z")
xlabel("Time lag from ADN")
show()

figure()
subplot(211)
plot(cc1.mean(1), label = 'lmn')
plot(cc2.mean(1), label = 'adn')
legend()
title("Event corr")
xlabel("Down center")
subplot(212)
plot(stas2.mean(1), label = 'lmn')
plot(stas.mean(1), label = 'adn')
legend()
xlabel("Down center")
title("Reactivation")
show()


sys.exit()

for e in allcc.keys():
    allcc[e] = pd.concat(allcc[e], 1)
    allcc[e] = allcc[e].apply(zscore)

datatosave = {'allcc':allcc}
cPickle.dump(datatosave, open(os.path.join('/home/guillaume/Dropbox/CosyneData', 'MUA_LMN_ADN.pickle'), 'wb'))



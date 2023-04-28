# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-14 18:02:55
import numpy as np
import pandas as pd
import pynapple as nap
import sys, os
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from functions import *
import pynacollada as pyna
from scipy.signal import filtfilt
from scipy.stats import zscore
from numba import jit

@jit(nopython=True)
def get_successive_bins(s, thr=-1.5):
    idx = np.zeros(len(s)-1)
    for i in range(len(s)-1):
        if s[i] > thr and s[i+1] > thr:
            idx[i] = 1
    return idx


############################################################################################### 
# GENERAL infos
###############################################################################################
# data_directory = '/mnt/DataGuillaume/'
data_directory = '/mnt/DataRAID2/'
# data_directory = '/media/guillaume/LaCie'


datasets = np.genfromtxt('/mnt/DataRAID2/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
# On Razer
# datasets = np.genfromtxt('/media/guillaume/LaCie/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')


datatosave = {}

# for s in datasets:
for s in ['LMN-PSB/A3019/A3019-220701A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes.getby_threshold('rate', 0.4)
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    angle = position['ry']
    tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    spikes.set_info(SI)
    r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
    spikes.set_info(halfr = r)

    psb = spikes.getby_category("location")['psb'].getby_threshold('SI', 1.0).getby_threshold('halfr', 0.5).index
    lmn = spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.2).getby_threshold('halfr', 0.5).index

    tokeep = np.hstack((psb, lmn))
    # spikes = spikes.getby_category("location")['psb']
    # spikes = spikes[tokeep]

    #########################
    # COUNT
    #########################    
    rates = {}
    for e, ep, bin_size, std in zip(
            ['wak', 'rem', 'sws'], 
            [wake_ep, rem_ep, sws_ep], 
            [0.2, 0.2, 0.01], 
            [1, 1, 1]
            ):
        count = spikes[tokeep].count(bin_size, ep)
        rate = count/bin_size
        rate = rate.as_dataframe()
        #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
        rate = rate.apply(zscore)
        rates[e] = rate        

    #########################
    # COSINE SIMILARITY
    #########################
    sim = {}
    for k in rates.keys():
        r = rates[k][psb].values        
        nrm = np.linalg.norm(r, axis=1)
        sim[k] = np.sum(r[0:-1]*r[1:], 1)/(nrm[0:-1]*nrm[1:])

    idxs = {}
    for k in sim.keys():
        s = rates[k].mean(1).values
        idxs[k] = get_successive_bins(s, thr=0)


    figure()
    for i,k in enumerate(sim.keys()):
        subplot(3,1,i+1)
        hist(sim[k][idxs[k]==1], np.linspace(-1, 1, 20))
        xlim(-1, 1)
        title(k)
    show()

    ##############################
    # REACTIVATION
    ##############################
    rate = rates['wak'][lmn]
    C = (1/rate.shape[0])*np.dot(rate.values.T, rate.values)
    C[np.diag_indices_from(C)] = 0.0

    rate = rates['sws'][lmn]
    p = np.sum(np.dot(rate.values, C) * rate.values, 1)

    rea = (p[0:-1] + p[1:])/2

    figure()
    scatter(rea[idxs["sws"] == 1], sim['sws'][idxs['sws']==1])


    

    sys.exit()
    
    # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
    pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))
    pairs = pd.MultiIndex.from_tuples(pairs)
    r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

    for ep in rates.keys():
        tmp = np.corrcoef(rates[ep].values.T)
        r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

    name = data.basename    
    pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
    pairs = pd.MultiIndex.from_tuples(pairs)
    r.index = pairs



    sys.exit()

    # MUA    
    # binsize = 0.1
    # total = spikes.count(binsize, sws_ep).sum(1)/binsize
    # total2 = total.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
    # total2 = nap.Tsd(total2, time_support = sws_ep)
    # total2 = total2 - total2.mean()
    # total2 = total2 / total2.std()
    # nrem3_ep = total2.threshold(np.percentile(total2, 50), method='below').time_support
    # nrem3_ep = nrem3_ep.merge_close_intervals(binsize*2)

    # # sta2_ep = sta2_ep.drop_long_intervals(2)
    # nrem2_ep = sws_ep.set_diff(nrem3_ep)

    # # nrem2_ep = nrem2_ep.drop_short_intervals(0.3)
    # # nrem3_ep = nrem3_ep.drop_short_intervals(0.3)

    # # nrem2_ep = nrem2_ep.drop_long_intervals(10)
    # # nrem3_ep = nrem3_ep.drop_long_intervals(5)


    # # [durations[2].append(v) for v in nrem2_ep['end'] - nrem2_ep['start']]
    # # [durations[3].append(v) for v in nrem3_ep['end'] - nrem3_ep['start']]


    
    # LFP
    frequency = 1250.0
    binsize = 0.2
    lfp = data.load_lfp(channel=0,extension='.eeg',frequency=1250)
    lfp = lfp.restrict(sws_ep)
    signal = pyna.eeg_processing.bandpass_filter(lfp, 4.0, 9.0, 1250, order=2)

    power = signal.pow(2).bin_average(binsize)
    power = power.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=10)
    power = power - power.mean()
    power = power / power.std()
    power = nap.Tsd(t=power.index.values, 
        d = power.values,
        time_support = sws_ep)

    nrem3_ep = power.threshold(np.percentile(power, 50)).time_support
    nrem3_ep = nrem3_ep.merge_close_intervals(0.1)    
    # # sta2_ep = sta2_ep.drop_long_intervals(2)
    nrem2_ep = sws_ep.set_diff(nrem3_ep)

    nrem3_ep = nrem3_ep.drop_short_intervals(0.1)
    nrem2_ep = nrem2_ep.drop_short_intervals(0.1)

    datatosave[s] = power

    # c3 = spikes.count(binsize).restrict(nrem3_ep).sum()/nrem3_ep.tot_length()
    # c2 = spikes.count(binsize).restrict(nrem2_ep).sum()/nrem2_ep.tot_length()

    # c3 = c3.sort_values()
    # c2 = c2[c3.index]s

    # bar(np.arange(len(c3)), c3-c2)
    # ylabel("Stage3 - Stage 2")




    # mua = total2 - total2.min()
    # mua = mua / mua.max()

    # power = power - power.min()
    # power = power / power.max()

    # figure()
    # ax = subplot(211)                                          
    # plot(lfp)                                                  
    # plot(signal)
    # subplot(212, sharex=ax)
    # plot(power, label = 'power')
    # plot(mua, label = 'mua')
    # legend()

    # figure()
    # loglog(mua.values, power.values, '.', alpha=0.2)
    # xlabel("MUA")
    # ylabel("POWER DELTA")
    # title(pearsonr(np.log(mua.values+0.001), np.log(power.values+0.001)))

    data.write_neuroscope_intervals('.nrem2.evt', nrem2_ep, 'PyNREM2')
    data.write_neuroscope_intervals('.nrem3.evt', nrem3_ep, 'PyNREM3')    
    
    # data.write_neuroscope_intervals('.nrem2.evt', nrem2_ep, 'PyNREM2')
    # data.write_neuroscope_intervals('.nrem3.evt', nrem3_ep, 'PyNREM3')    


# cPickle.dump(datatosave, open(
#     os.path.join('/home/guillaume/Dropbox/CosyneData', 'DELTA_POWER_PSB.pickle'), 'wb'
#     ))



# for k in durations:
#     durations[k] = np.array(durations[k])

# figure()
# subplot(211)
# hist(durations[2], 50)
# subplot(212)
# hist(durations[3], 50)
# show()



# # figure()
# # ax = subplot(211)
# # for n in spikes.index:
# #     plot(spikes[n].restrict(sws_ep).fillna(n), '|')
# # subplot(212, sharex =ax)
# # plot(total2.restrict(sws_ep))
# # plot(total2.restrict(sta2_ep), '.', color = 'blue')
# # show()

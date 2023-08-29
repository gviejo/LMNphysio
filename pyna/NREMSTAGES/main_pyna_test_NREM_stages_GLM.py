# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-06-14 16:45:11
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-03-09 16:08:09
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from scipy.linalg import hankel

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
# data_directory = '/mnt/DataGuillaume/'
data_directory = '/mnt/DataRAID2/'
# data_directory = '/media/guillaume/LaCie'


datasets = np.genfromtxt('/mnt/DataRAID2/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')
# On Razer
# datasets = np.genfromtxt('/media/guillaume/LaCie/datasets_LMN_PSB.list', delimiter = '\n', dtype = str, comments = '#')

datatosave = {}

allr = []
pearson = {}

coefs_mua = {e:[] for e in ['wak', 'nrem2', 'nrem3']}
coefs_pai = {e:[] for e in ['wak', 'nrem2', 'nrem3']}

pairs_info = pd.DataFrame(columns = ['offset', 'session'])



for s in datasets:
# for s in ['LMN-PSB/A3019/A3019-220701A']:
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
    angle = position['ry']
    tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
    spikes.set_info(SI)
    r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
    spikes.set_info(halfr = r)
    tcurves = tuning_curves
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    psb = spikes.getby_category("location")['psb']
    spikes = spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.2).getby_threshold('halfr', 0.5)
    
    try:
        maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
        
    except:
        meanwf, maxch = data.load_mean_waveforms(spike_count=1000)
        maxch.to_csv(data.nwb_path + "/maxch.csv")        


    lmn = spikes.index
    #################################################################################################
    #DETECTION STAGE 1/ STAGE 2 States
    #################################################################################################

    # # MUA    
    # binsize = 0.1
    # total = spikes.count(binsize, sws_ep)
    # total = total.bin_average(binsize).sum(1)
    # total2 = total.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=10)
    # total2 = nap.Tsd(total2, time_support = sws_ep)
    # total2 = total2 - total2.mean()
    # total2 = total2 / total2.std()
    # power = total2    
    # nrem2_ep = total2.threshold(0).time_support

    # nrem3_ep = total2.threshold(0, method='below').time_support
    # # nrem2_ep = nrem2_ep.merge_close_intervals(binsize*2)

    # # sta2_ep = sta2_ep.drop_long_intervals(2)
    # # nrem3_ep = sws_ep.set_diff(nrem2_ep)

    # nrem2_ep = nrem2_ep.drop_short_intervals(0.1)
    # nrem3_ep = nrem3_ep.drop_short_intervals(0.1)

    # # nrem2_ep = nrem2_ep.drop_long_intervals(10)
    # # nrem3_ep = nrem3_ep.drop_long_intervals(5)


    # # [durations[2].append(v) for v in nrem2_ep['end'] - nrem2_ep['start']]
    # # [durations[3].append(v) for v in nrem3_ep['end'] - nrem3_ep['start']]


    #############################
    # # LFP
    frequency = 1250.0
    binsize = 0.02
    lfp = data.load_lfp(channel=16,extension='.eeg',frequency=1250)
    
    # lfp = downsample(lfp, 1, 5)
    # signal = pyna.eeg_processing.bandpass_filter(lfp, 0.5, 4.0, 250, order=3)
    # lfp = lfp.restrict(sws_ep)
    # signal = signal.restrict(sws_ep)
    
    signal = pyna.eeg_processing.bandpass_filter(lfp, 60.0, 100.0, 1250, order=3)
    signal = signal.restrict(sws_ep)
    lfp = lfp.restrict(sws_ep)

    power = signal.pow(2).bin_average(binsize)
    power = power.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=20)
    power = power - power.mean()
    power = power / power.std()
    power = nap.Tsd(t=power.index.values, 
        d = power.values,
        time_support = sws_ep)

    nrem2_ep = power.threshold(np.percentile(power, 70)).time_support
    
    nrem3_ep = power.threshold(np.percentile(power, 30), 'below').time_support

    nrem2_ep = nrem2_ep.drop_short_intervals(0.4)
    nrem3_ep = nrem3_ep.drop_short_intervals(0.4)

    ###############################

    ############################################################################################### 
    # LMN GLM CORRELATION
    ###############################################################################################    
    pairs = list(combinations(lmn, 2))

    # pairs = [p for p in pairs if np.abs(maxch[p[0]] - maxch[p[1]]) > 1]

    r = {'wak':{},'nrem2':{},'nrem3':{}}
    for i, p in enumerate(pairs):
        tar_neuron = p[0]
        reg_neuron = p[1]
        pair_name = data.basename + '_' + str(p[0]) + '_' + str(p[1])
        a = peaks[tar_neuron] - peaks[reg_neuron]
        pair_offset = np.abs(np.arctan2(np.sin(a), np.cos(a)))        
        pairs_info.loc[pair_name, 'offset'] = pair_offset
        pairs_info.loc[pair_name, 'session'] = s

        ## MUA ########
        group = list(set(lmn) - set([reg_neuron]))
        mua = {0:nap.Ts(t=np.sort(np.hstack([spikes[j].index.values for j in group])))}
        mua = nap.TsGroup(mua, time_support = spikes.time_support, location = np.array(['lmn']))
    
        for e, ep, binsize, windowsize in zip(
                ['wak', 'nrem2', 'nrem3'], 
                [wake_ep, nrem2_ep, nrem3_ep], 
                [0.2, 0.01, 0.01], 
                [1.0, 0.1, 0.1]
                ):

            count = mua[0].count(binsize, ep)
            count = nap.TsdFrame(
                t = count.index.values,
                d = np.atleast_2d(count.values).T,
                c = [0]
                )
            
            rate = count/binsize
            # rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            rate = StandardScaler().fit_transform(rate.values)            
            mua_offset, time_idx = offset_matrix(rate.flatten(), binsize, windowsize)
            
            count = pd.concat([
                spikes[n].count(binsize, ep) for n in list(p)
                ], 1)

            # rate = count.values/binsize
            rate = count/binsize
            #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
            rate = StandardScaler().fit_transform(rate.values)
            neuron_offset, time_idx = offset_matrix(rate[:,1], binsize, windowsize)

            offset_tmp = np.hstack((mua_offset, neuron_offset))

            # Version grouped offset            
            glm = PoissonRegressor(max_iter = 100)                    
            glm.fit(offset_tmp, count.values[:,0])

            coefs_mua[e].append(pd.DataFrame(
                index=time_idx, data=glm.coef_[0:len(time_idx)], columns=[pair_name]))
            coefs_pai[e].append(pd.DataFrame(
                index=time_idx, data=glm.coef_[len(time_idx):], columns=[pair_name]))

            r[e][pair_name] = glm.coef_[len(time_idx) :][time_idx == 0][0]

    allr.append(pd.DataFrame(r))
    #######################
    # COMPUTING PEARSON R FOR EACH SESSION
    #######################
    r = pd.DataFrame(r)
    pearson[s] = np.zeros((3))
    pearson[s][0] = scipy.stats.pearsonr(r['wak'], r['nrem2'])[0]
    pearson[s][1] = scipy.stats.pearsonr(r['wak'], r['nrem3'])[0]
    pearson[s][2] = len(spikes)


    

allr = pd.concat(allr, 0)

pearson = pd.DataFrame(pearson).T
pearson.columns = ['nrem2', 'nrem3', 'count']


pairs_info = pairs_info.sort_values(by="offset")
for k in coefs_mua.keys():
    coefs_mua[k] = pd.concat(coefs_mua[k], 1)
    coefs_pai[k] = pd.concat(coefs_pai[k], 1)        
    coefs_mua[k] = coefs_mua[k][pairs_info.index]
    coefs_pai[k] = coefs_pai[k][pairs_info.index]


figure()
gs = GridSpec(3,3)

inters = np.linspace(0, np.pi, 4)
idx = np.digitize(pairs_info['offset'], inters)-1

for j, k in enumerate(['wak', 'nrem2', 'nrem3']):
    subplot(gs[0,j])
    tmp = coefs_pai[k]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
    tmp = tmp.values.T
    imshow(tmp, aspect='auto', cmap = 'jet')
    title(k)

    subplot(gs[1,j])
    for l in range(3):
        tmp = coefs_pai[k].iloc[:,idx==l]#.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        plot(tmp.mean(1), '-')            



figure()
subplot(131)
plot(allr['wak'], allr['nrem2'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['nrem2'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
plot(x, x*m + b)
ylim(allr[["nrem2", "nrem3"]].min().min(),allr[["nrem2", "nrem3"]].max().max())
xlim(allr["wak"].min(), allr["wak"].max())
xlabel('wake')
ylabel('nrem2')
r, p = scipy.stats.pearsonr(allr['wak'], allr['nrem2'])
title('r = '+str(np.round(r, 3)))

subplot(132)
plot(allr['wak'], allr['nrem3'], 'o',  alpha = 0.5)
m, b = np.polyfit(allr['wak'].values, allr['nrem3'].values, 1)
x = np.linspace(allr['wak'].min(), allr['wak'].max(), 4)
plot(x, x*m + b)
ylim(allr[["nrem2", "nrem3"]].min().min(),allr[["nrem2", "nrem3"]].max().max())
xlim(allr["wak"].min(), allr["wak"].max())
xlabel('wake')
ylabel('nrem3')
r, p = scipy.stats.pearsonr(allr['wak'], allr['nrem3'])
title('r = '+str(np.round(r, 3)))

subplot(133)
y = pearson
for j, e in enumerate(['nrem2', 'nrem3']):
    plot(np.ones(len(y))*j + np.random.randn(len(y))*0.1, y[e], 'o', markersize=2)
    plot([j-0.2, j+0.2], [y[e].mean(), y[e].mean()], '-', linewidth=0.75)
xticks([0, 1])
xlim(-0.4,1.4)
ylim(0, 1)
# print(scipy.stats.ttest_ind(y["rem"], y["sws"]))
print(scipy.stats.wilcoxon(y["nrem2"], y["nrem3"]))



show()


# figure()

# sws2_ep = sws_ep.loc[[(sws_ep["end"] - sws_ep["start"]).sort_values().index[-1]]]

# # sws2_ep = nap.IntervalSet(
# #     start = 5652, end = 5665
# #     )

# ax = subplot(311)

# for s, e in nrem2_ep.intersect(sws2_ep).values:
#     axvspan(s, e, color = 'green', alpha=0.1)
# for s, e in nrem3_ep.intersect(sws2_ep).values:
#     axvspan(s, e, color = 'orange', alpha=0.1)  

# axvline(5652)

# plot(power.restrict(sws2_ep))

# subplot(312, sharex = ax)
# plot(lfp.restrict(sws2_ep))
# plot(signal.restrict(sws2_ep))

# subplot(313, sharex = ax)
# for i,n in enumerate(peaks[lmn].sort_values().index.values):
#     plot(spikes[n].restrict(sws2_ep).fillna(i), '|', 
#         markersize = 10, markeredgewidth=1)

# show()



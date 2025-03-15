# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-29 13:46:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-03-15 18:37:36
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
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
import yaml
import warnings
warnings.simplefilter('ignore', category=UserWarning)




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

datasets = yaml.safe_load(
    open(os.path.join(
        data_directory,
        "datasets_OPTO.yaml"), "r"))['opto']


SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.0
    }

allr = {}
corr = {}
allfr = {}
alltc = {}

# allfr = {"wake":[], "sleep":[]}
# allmeta = {"wake":[], "sleep":[]}
# alltc = {"wake":[], "sleep":[]}

for st in ['adn', 'lmn']:

    allr[st] = {}
    corr[st] = {}
    allfr[st] = {}
    alltc[st] = {}

    for gr in ['opto', 'ctrl']:    

        allr[st][gr] = {}
        corr[st][gr] = {}
        allfr[st][gr] ={}
        alltc[st][gr] = {}

        for sd in ['ipsi', 'bilateral']:

            try:
                dataset = datasets[st][gr][sd]['wake']
            except:
                dataset = []

            if len(dataset):


                allr[st][gr][sd] = []
                corr[st][gr][sd] = []
                allfr[st][gr][sd] = []
                alltc[st][gr][sd] = []
                alltc[st][gr][sd] = {'opto':[], 'pre':[]}        

                for s in dataset:
                    
                    ############################################################################################### 
                    # LOADING DATA
                    ###############################################################################################
                    path = os.path.join(data_directory, "OPTO", s)
                    basename = os.path.basename(path)
                    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
                    
                    if not os.path.exists(filepath):
                        print("missing ", st, gr, s)
                        sys.exit()
                    else:

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
                        opto_ep = nwb["opto"]
                        
                        
                            
                    spikes = spikes[spikes.location == st]
                    stim_duration = 10.0
                    opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start'])>=stim_duration-0.001]
                    

                    ############################################################################################### 
                    # COMPUTING TUNING CURVES
                    ###############################################################################################
                    # Opto ep is always the second wake
                    assert len(opto_ep.intersect(position.time_support[0])) == 0

                    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
                    tuning_curves = smoothAngularTuningCurves(tuning_curves)
                    tcurves = tuning_curves
                    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
                    spikes.set_info(SI)
                    spikes.set_info(max_fr = tcurves.max())
                    
                    # figure()
                    # for i in range(len(spikes)):
                    #     subplot(6,6,i+1, projection='polar')
                    #     plot(tuning_curves[spikes.keys()[i]])
                    #     title(np.round(spikes.SI[spikes.keys()[i]], 3))
                    # show()

                    spikes = spikes.getby_threshold("SI", SI_thr[st])
                    spikes = spikes.getby_threshold("rate", 1.0)
                    spikes = spikes.getby_threshold("max_fr", 3.0)

                    tokeep = spikes.index
                    tcurves = tcurves[tokeep]
                    tuning_curves = tcurves
                    peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
                    order = np.argsort(peaks.values)
                    spikes.set_info(order=order, peaks=peaks)
                    tuning_curves.columns = [basename+"_"+str(i) for i in tuning_curves.columns]

                    ############################################################################################### 
                    # COMPUTING OPTO TUNING CURVES
                    ###############################################################################################
                    opto_tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = opto_ep)
                    opto_tuning_curves = smoothAngularTuningCurves(opto_tuning_curves)            
                    SI_opto = nap.compute_1d_mutual_info(opto_tuning_curves, position['ry'], opto_ep, (0, 2*np.pi))

                    opto_tuning_curves.columns = [basename+"_"+str(i) for i in opto_tuning_curves.columns]
                    
                    ############################################################################################### 
                    # FIRING RATE MODULATION
                    ###############################################################################################    

                    # peth = nap.compute_perievent(spikes[tokeep], nap.Ts(opto_ep.start), minmax=(-stim_duration, 2*stim_duration))
                    # frates = pd.DataFrame({n:np.sum(peth[n].count(0.05), 1).values for n in peth.keys()})

                    frates = nap.compute_eventcorrelogram(spikes, nap.Ts(opto_ep.start), stim_duration/20., stim_duration*2, norm=True)

                    frates.columns = [basename+"_"+str(i) for i in frates.columns]

                    ############################################################################################### 
                    # PEARSON CORRELATION
                    ###############################################################################################        
                    if len(tokeep) > 2:
                        
                        velocity = np.abs(computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2))
                        newwake_ep = velocity.threshold(0.02).time_support.drop_short_intervals(1).merge_close_intervals(1)

                        pre_ep = nap.IntervalSet(start=opto_ep.start-stim_duration, end=opto_ep.start)
                        # pre_ep = pre_ep.intersect(sws_ep)
                        pre_ep = pre_ep.drop_short_intervals(stim_duration-0.001)

                        rates = {}
                        for e, iset, bin_size, std in zip(['wak', 'pre', 'opto'], [newwake_ep, pre_ep, opto_ep], [0.3, 0.3, 0.3], [2, 2, 2]):
                            count = spikes.count(bin_size, iset)
                            rate = count/bin_size
                            rate = rate.smooth(std=bin_size*std).as_dataframe()
                            rate = rate.apply(zscore)                    
                            rates[e] = rate
                        
                        pairs = [basename+"_"+i+"-"+j for i,j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
                        r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

                        for e in rates.keys():
                            tmp = np.corrcoef(rates[e].values.T)
                            if len(tmp):
                                r[e] = tmp[np.triu_indices(tmp.shape[0], 1)]

                        allr[st][gr][sd].append(r)

                        #######################
                        # Session correlation
                        #######################

                        tmp = pd.DataFrame(index=[basename])
                        tmp['pre'] = scipy.stats.pearsonr(r['wak'], r['pre'])[0]
                        tmp['opto'] = scipy.stats.pearsonr(r['wak'], r['opto'])[0]
                        tmp['n'] = len(spikes)
                        corr[st][gr][sd].append(tmp)

                    
                    #######################
                    # SAVING
                    #######################
                    allfr[st][gr][sd].append(frates)
                    alltc[st][gr][sd]['opto'].append(opto_tuning_curves)
                    alltc[st][gr][sd]['pre'].append(tuning_curves)
                    
                    # metadata = spikes._metadata
                    # metadata.index = frates.columns
                    # allmeta[ep].append(metadata)
                    # tuning_curves.columns = frates.columns
                    # alltc[ep].append(tuning_curves)       

                 
                allr[st][gr][sd] = pd.concat(allr[st][gr][sd], axis=0)
                corr[st][gr][sd] = pd.concat(corr[st][gr][sd], axis=0)
                allfr[st][gr][sd] = pd.concat(allfr[st][gr][sd], axis=1)

                alltc[st][gr][sd]['opto'] = pd.concat(alltc[st][gr][sd]['opto'], axis=1)
                alltc[st][gr][sd]['pre'] = pd.concat(alltc[st][gr][sd]['pre'], axis=1)

                # allmeta[ep] = pd.concat(allmeta[ep], axis=0)
                # alltc[ep] = pd.concat(alltc[ep], axis=1)




change_fr = {}
for st in allfr.keys():
    change_fr[st] = {}
    for gr in allfr[st].keys():
        change_fr[st][gr] = pd.concat([
            allfr[st][gr].loc[-0.8:0].mean(),
            allfr[st][gr].loc[0:1].mean()], axis=1)
        change_fr[st][gr].columns = ['pre', 'opto']
        




figure()
gs = GridSpec(2, 2)
for i, st in zip(range(2), ['adn', 'lmn']):
# order = allmeta[ep].sort_values(by="SI").index.values
    # tmp = allfr[ep][order].loc[sl]    
    tmp = allfr[st]['opto'].loc[-10:20]
    tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1, mode='constant'))
    subplot(gs[0,i])
    plot(tmp, color = 'grey', alpha=0.2)
    plot(tmp.mean(1), color = 'blue')
    axvline(0)
    axvline(10)
    xlim(-10, 20)
    title(st)
    subplot(gs[1,i])    
    # tmp = tmp - tmp.loc[msl].mean(0)
    # tmp = tmp / tmp.std(0)    
    imshow(tmp.values.T, cmap = 'jet', aspect='auto')
    # title(ep)
tight_layout()


figure()
gs = GridSpec(2, 2)
for i, st in zip(range(2), ['adn', 'lmn']):
    tc1 = centerTuningCurves(alltc[st]['opto']['pre'])
    tc2 = centerTuningCurves(alltc[st]['opto']['opto'])
    order = tc1.loc[0].sort_values().index
    subplot(gs[i,0])
    imshow(tc1[order].values.T, aspect='auto')
    ylabel(st)
    subplot(gs[i,1])
    imshow(tc1[order].values.T, aspect='auto')



# savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_LMN_OPTO.png"))

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


figure(figsize=(26, 18))
gs = GridSpec(1, 2)
for i, st in enumerate(['adn', 'lmn']):
    gs2 = GridSpecFromSubplotSpec(3, 2, gs[0,i])

    # Opto vs ctrl
    subplot(gs2[0,0])
    e = 'opto'
    for j, gr in enumerate(['opto', 'ctrl']):

        r, p = scipy.stats.pearsonr(allr[st][gr]['wak'], allr[st][gr][e]) # Wak vs opto

        plot(allr[st][gr]['wak'], allr[st][gr][e], 'o', color = cycle[j], alpha = 0.5, label = 'r = '+str(np.round(r, 3)))

        m, b = np.polyfit(allr[st][gr]['wak'].values, allr[st][gr][e].values, 1)
        x = np.linspace(allr[st][gr]['wak'].min(), allr[st][gr]['wak'].max(),5)
        plot(x, x*m + b)
        xlabel('wak')
        ylabel(e)
        xlim(allr[st][gr]['wak'].min(), allr[st][gr]['wak'].max())
        # ylim(allr[st].iloc[:,1:].min().min(), allr[st].iloc[:,1:].max().max())
        ylim(allr[st][gr]['wak'].min(), allr[st][gr]['wak'].max())
        legend()
        title(f"{st.upper()} "+" chrimson vs tdtomato")
            
    # Opto vs preopto
    subplot(gs2[0,1])    
    gr = 'opto'
    for k, e in enumerate(['opto', 'pre']):

        r, p = scipy.stats.pearsonr(allr[st][gr]['wak'], allr[st][gr][e])

        plot(allr[st][gr]['wak'], allr[st][gr][e], 'o', color = cycle[k], alpha = 0.5, label = 'r = '+str(np.round(r, 3)))

        m, b = np.polyfit(allr[st][gr]['wak'].values, allr[st][gr][e].values, 1)
        x = np.linspace(allr[st][gr]['wak'].min(), allr[st][gr]['wak'].max(),5)
        plot(x, x*m + b)
        xlabel('wak')
        ylabel(e)
        xlim(allr[st][gr]['wak'].min(), allr[st][gr]['wak'].max())
        # ylim(allr[st].iloc[:,1:].min().min(), allr[st].iloc[:,1:].max().max())
        ylim(allr[st][gr]['wak'].min(), allr[st][gr]['wak'].max())
        legend()
        title(f"{st.upper()} "+" chrimson light vs no light")

    # Pearson per sesion
    subplot(gs2[1,0])
    
    corr2 = corr[st]['opto']
    corr2 = corr2[corr2['n']>3]
    for j, e in enumerate(['opto', 'pre']):
        plot(np.random.randn(len(corr2))*0.1+np.ones(len(corr2))*j, corr2[e], 'o', markersize=6)

    corr2 = corr[st]['ctrl']
    corr2 = corr2[corr2['n']>3]
    plot(np.random.randn(len(corr2))*0.1+np.ones(len(corr2))*2, corr2[e], 'o', markersize=6)
    ylim(-1, 1)
    xticks([0, 1, 2], ['chrimson', 'pre-opto\nchrimson', 'tdtomate\nctrl'])

    ylabel("Pearson r")
    title(st.upper())
    
    # Firing rate
    subplot(gs2[1,1])    
    for j, gr in enumerate(['opto', 'ctrl']):
        chfr = change_fr[st][gr]
        delta = (chfr['opto'] - chfr['pre'])/chfr['pre']
        plot(np.random.randn(len(delta))*0.076+np.ones(len(chfr))*j, delta, 'o', markersize=6)        
    title(st.upper())
    ylabel("Delta firing rate")
    ylim(-1, 1)
    xticks([0, 1], ['chrimson', 'tdtomato\nctrl'])

    # Tuning curve
    for j, e in enumerate(['opto', 'pre']):
        subplot(gs2[2,j])
        tc = centerTuningCurves(alltc[st]['opto'][e])
        tc = tc/tc.loc[0]

        plot(tc.mean(1))
        m = tc.mean(1)
        s = tc.std(1)
        fill_between(m.index.values, m.values-s.values, m.values+s.values, alpha=0.1)
        title(e)




gs.update(right=0.98, left=0.1)
# savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_OPTO_WAKE.png"))
show()


# ##################################################################
# # FOR FIGURE 1
# ##################################################################
# datatosave = {
#     "allfr":allfr,
#     "allmeta":allmeta,
#     "alltc":alltc
# }


# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# file_name = "OPTO_PSB.pickle"

# import _pickle as cPickle

# with open(os.path.join(dropbox_path, file_name), "wb") as f:
#     cPickle.dump(datatosave, f)
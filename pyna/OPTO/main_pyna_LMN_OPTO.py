# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-29 13:46:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-07-26 18:33:19
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

datasets = yaml.safe_load(open("/mnt/ceph/users/gviejo/datasets_OPTO.yaml", "r"))
datasets = datasets['opto']['opto_lmn_psb']

SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.0
    }

allr = []
corr = []
allfr = {"wake":[], "sleep":[]}
allmeta = {"wake":[], "sleep":[]}
alltc = {"wake":[], "sleep":[]}

for ep in ['wake', 'sleep']:
    for s in datasets[ep]:
        print(ep, s)    
        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        path = os.path.join(data_directory, "OPTO", s)
        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake'].loc[[0]]
        sleep_ep = data.epochs["sleep"]
        try:
            sws_ep = data.read_neuroscope_intervals('sws')
        except:
            sws_ep = nap.IntervalSet([], [])
        # rem_ep = data.read_neuroscope_intervals('rem')
        # down_ep = data.read_neuroscope_intervals('down')
        spikes = spikes.getby_threshold("rate", 1.0)
        idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
        spikes = spikes[idx]

        ############################################################################################### 
        # LOADING OPTO INFO
        ###############################################################################################            
        try:
            os.remove(os.path.join(path, os.path.basename(path))+"_opto_ep.npz")
            opto_ep = nap.load_file(os.path.join(path, os.path.basename(path))+"_opto_ep.npz")
        except:
            opto_ep = []
            epoch = 0
            while len(opto_ep) == 0:
                try:
                    opto_ep = loadOptoEp(path, epoch=epoch, n_channels = 2, channel = 0)
                    opto_ep = opto_ep.intersect(data.epochs[ep])
                except:
                    pass                    
                epoch += 1
                if epoch == 10:
                    sys.exit()
            opto_ep.save(os.path.join(path, os.path.basename(path))+"_opto_ep")

        if ep == "sleep":
            opto_ep = opto_ep.intersect(sws_ep)


        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
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

        spikes = spikes.getby_threshold("SI", SI_thr["lmn"])
        spikes = spikes.getby_threshold("rate", 1.0)
        spikes = spikes.getby_threshold("max_fr", 3.0)

        tokeep = spikes.index
        tcurves = tcurves[tokeep]
        tuning_curves = tcurves
        peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        order = np.argsort(peaks.values)
        spikes.set_info(order=order, peaks=peaks)


        ############################################################################################### 
        # FIRING RATE MODULATION
        ###############################################################################################    
        if ep == "wake":
            stim_duration = 10.0
        if ep == "sleep":
            stim_duration = 1.0

        opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start'])>=stim_duration-0.001]

        # peth = nap.compute_perievent(spikes[tokeep], nap.Ts(opto_ep.start), minmax=(-stim_duration, 2*stim_duration))
        # frates = pd.DataFrame({n:np.sum(peth[n].count(0.05), 1).values for n in peth.keys()})

        frates = nap.compute_eventcorrelogram(spikes[tokeep], nap.Ts(opto_ep.start), stim_duration/20., stim_duration*2, norm=True)

        frates.columns = [data.basename+"_"+str(i) for i in frates.columns]

        ############################################################################################### 
        # PEARSON CORRELATION
        ###############################################################################################        
        if len(tokeep) > 2 and ep == "sleep":
            
            velocity = np.abs(computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2))
            newwake_ep = velocity.threshold(0.02).time_support.drop_short_intervals(1).merge_close_intervals(1)

            rates = {}
            sws2_ep = sws_ep.intersect(sleep_ep.loc[[0]])

            for e, iset, bin_size, std in zip(['wak', 'sws', 'opto'], [newwake_ep, sws2_ep, opto_ep], [0.3, 0.03, 0.03], [1, 1, 1]):
                count = spikes.count(bin_size, iset)
                rate = count/bin_size
                rate = rate.as_dataframe()
                rate = rate.apply(lambda x: gaussian_filter1d(x, sigma=std, mode='constant'))
                rate = rate.apply(zscore)                    
                rates[e] = nap.TsdFrame(rate)            
            
            pairs = [data.basename+"_"+i+"-"+j for i,j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
            r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

            for e in rates.keys():
                tmp = np.corrcoef(rates[e].values.T)
                if len(tmp):
                    r[e] = tmp[np.triu_indices(tmp.shape[0], 1)]

            allr.append(r)

            #######################
            # Session correlation
            #######################

            tmp = pd.DataFrame(index=[data.basename])
            tmp['sws'] = scipy.stats.pearsonr(r['wak'], r['sws'])[0]
            tmp['opto'] = scipy.stats.pearsonr(r['wak'], r['opto'])[0]
            tmp['n'] = len(spikes)
            corr.append(tmp)        

        #######################
        # SAVING
        #######################
        allfr[ep].append(frates)
        metadata = spikes._metadata
        metadata.index = frates.columns
        allmeta[ep].append(metadata)
        tuning_curves.columns = frates.columns
        alltc[ep].append(tuning_curves)       

                        
    allfr[ep] = pd.concat(allfr[ep], 1)
    allmeta[ep] = pd.concat(allmeta[ep], 0)
    alltc[ep] = pd.concat(alltc[ep], 1)


allr = pd.concat(allr, 0)
corr = pd.concat(corr, 0)



figure()
gs = GridSpec(2, 2)
for i, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(-2,12), slice(-1,2)], [(0, 10), (0,1)]):
    order = allmeta[ep].sort_values(by="SI").index.values
    tmp = allfr[ep][order].loc[sl]    
    tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1, mode='constant'))
    subplot(gs[0,i])
    plot(tmp, color = 'grey', alpha=0.2)
    plot(tmp.mean(1), color = 'blue')
    axvline(msl[0])
    axvline(msl[1])
    xlim(sl.start, sl.stop)
    title(ep)
    subplot(gs[1,i])    
    # tmp = tmp - tmp.loc[msl].mean(0)
    # tmp = tmp / tmp.std(0)    
    imshow(tmp.values.T, cmap = 'jet', aspect='auto')
    title(ep)
tight_layout()

savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_LMN_OPTO.png"))



figure()
epochs = ['sws', 'opto']
gs = GridSpec(2, len(epochs))
for i, e in enumerate(epochs):
    subplot(gs[0,i])
    plot(allr['wak'], allr[e], 'o', color = 'red', alpha = 0.5)
    m, b = np.polyfit(allr['wak'].values, allr[e].values, 1)
    x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
    plot(x, x*m + b)
    xlabel('wak')
    ylabel(e)
    xlim(allr['wak'].min(), allr['wak'].max())
    ylim(allr.iloc[:,1:].min().min(), allr.iloc[:,1:].max().max())
    r, p = scipy.stats.pearsonr(allr['wak'], allr[e])
    title('r = '+str(np.round(r, 3)))

subplot(gs[1,0])
corr2 = corr[corr['n']>=5]
for i, e in enumerate(['sws', 'opto']):
    plot(np.random.randn(len(corr2))*0.1+np.ones(len(corr2))*i, corr2[e], 'o', markersize=6)
ylim(-1, 1)
xticks(np.arange(corr2.shape[1]), corr2.columns)
title(scipy.stats.wilcoxon(corr2['sws'], corr2['opto']))

tight_layout()

savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_LMN_OPTO_CORR.png"))
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
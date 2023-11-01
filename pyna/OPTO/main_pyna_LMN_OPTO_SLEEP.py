# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-29 13:46:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-10-31 17:12:26
import numpy as np
import pandas as pd
import pynapple as nap
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
# from sklearn.linear_model import PoissonRegressor



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

# data_directory = "/media/guillaume/My Passport"

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_OPTO_SLEEP.list'), delimiter = '\n', dtype = str, comments = '#')


SI_thr = {
    'adn':0.5, 
    'lmn':0.2,
    'psb':1.5
    }

allr = []
corr = []
allfr = []
allmeta = []
alltc = []

for s in datasets:
    print(s)    
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, 'OPTO', s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake'].loc[[0]]
    sleep_ep = data.epochs["sleep"]
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    # down_ep = data.read_neuroscope_intervals('down')


    idx = spikes._metadata[spikes._metadata["location"].str.contains("lmn")].index.values
    spikes = spikes[idx]
      
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves)    
    tcurves = tuning_curves
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
    spikes.set_info(SI)
    spikes.set_info(max_fr = tcurves.max())

    spikes = spikes.getby_threshold("SI", SI_thr["lmn"])
    spikes = spikes.getby_threshold("rate", 1.0)
    spikes = spikes.getby_threshold("max_fr", 3.0)

    tokeep = spikes.index
    tcurves = tcurves[tokeep]
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
    order = np.argsort(peaks.values)
    spikes.set_info(order=order, peaks=peaks)


    ############################################################################################### 
    # LOADING OPTO INFO
    ###############################################################################################    
    try:
        opto_ep = nap.load_file(os.path.join(path, os.path.basename(path))+"_opto_sleep_ep.npz")
    except:
        opto_ep = []
        epoch = 0
        while len(opto_ep) == 0:
            try:
                opto_ep = loadOptoEp(path, epoch=epoch, n_channels = 2, channel = 0)
                opto_ep = opto_ep.intersect(data.epochs["sleep"])
            except:
                pass                    
            epoch += 1
            if epoch == 10:
                sys.exit()
        opto_ep.save(os.path.join(path, os.path.basename(path))+"_opto_sleep_ep")


    ############################################################################################### 
    # FIRING RATE MODULATION
    ###############################################################################################    
    stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 6)

    # peth = nap.compute_perievent(spikes[tokeep], nap.Ts(opto_ep["start"].values), minmax=(-stim_duration, 2*stim_duration))
    # frates = pd.DataFrame({n:peth[n].count(0.05).sum(1) for n in peth.keys()})
    frates = nap.compute_eventcorrelogram(spikes[tokeep], nap.Ts(opto_ep["start"].values), 0.05, 2, norm=True)
    frates.columns = [data.basename+"_"+str(i) for i in frates.columns]    
        
    print(s, len(tokeep), stim_duration)

    if len(tokeep) > 2:


        # figure()
        # for i in range(len(tokeep)):
        #     subplot(4, 4, i+1, projection='polar')
        #     plot(tcurves[tokeep[i]])
        
        
        velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.05).time_support.drop_short_intervals(1).merge_close_intervals(1)


        ############################################################################################### 
        # PEARSON CORRELATION
        ###############################################################################################        
        rates = {}
        sws2_ep = sws_ep.intersect(sleep_ep.loc[[0]])        

        for e, ep, bin_size, std in zip(['wak', 'sws', 'opto'], [newwake_ep, sws_ep, opto_ep], [0.3, 0.03, 0.03], [1, 1, 1]):
            count = spikes.count(bin_size, ep)
            rate = count/bin_size
            rate = rate.as_dataframe()
            rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
            rate = rate.apply(zscore)                    
            rates[e] = nap.TsdFrame(rate)
        
        
        pairs = [data.basename+"_"+i+"-"+j for i,j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
        r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

        for ep in rates.keys():
            tmp = np.corrcoef(rates[ep].values.T)
            if len(tmp):
                r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

        # to_keep = []
        # for p in r.index:
        #     tmp = spikes._metadata.loc[np.array(p.split("_")[1].split("-"), dtype=np.int32), ['group', 'maxch']]
        #     if tmp['group'].iloc[0] == tmp['group'].iloc[1]:
        #         if tmp['maxch'].iloc[0] != tmp['maxch'].iloc[1]:
        #             to_keep.append(p)
        # r = r.loc[to_keep]
        
        #######################
        # Session correlation
        #######################

        tmp = pd.DataFrame(index=[data.basename])
        tmp['sws'] = scipy.stats.pearsonr(r['wak'], r['sws'])[0]
        tmp['opto'] = scipy.stats.pearsonr(r['wak'], r['opto'])[0]
        
        corr.append(tmp)
                    
        #######################
        # SAVING
        #######################
        allr.append(r)
        allfr.append(frates)
        metadata = spikes._metadata
        metadata.index = frates.columns
        allmeta.append(metadata)
        tcurves.columns = frates.columns
        alltc.append(tcurves)


allr = pd.concat(allr, 0)
corr = pd.concat(corr, 0)
allfr = pd.concat(allfr, 1)
allmeta = pd.concat(allmeta, 0)



print(scipy.stats.wilcoxon(corr.iloc[:,-2], corr.iloc[:,-1]))


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
for i, e in enumerate(corr.columns):
    plot(np.random.randn(len(corr))*0.1+np.ones(len(corr))*i, corr[e], 'o')
ylim(0, 1)
xticks(np.arange(corr.shape[1]), corr.columns)

subplot(gs[1,1])
plot(allfr.loc[-1:2], alpha = 0.2, color = 'grey')
plot(allfr.loc[-1:2].mean(1), alpha = 1.0, color = 'red')
show()


##################################################################
# FOR FIGURE 1
##################################################################

datatosave = {
    "corr":corr,
    "allr":allr,
    "allfr":allfr,
    "allmeta":allmeta,
    "alltc":alltc
}


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
file_name = "OPTO_LMN_sleep.pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)
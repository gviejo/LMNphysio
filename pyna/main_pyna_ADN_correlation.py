# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 19:20:07
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-09-16 16:57:00

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
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
if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)

allr = []
pearson = {}

for s in datasets:
# for s in ['LMN-ADN/A5043/A5043-230301A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    if os.path.isdir(os.path.join(path, "pynapplenwb")):
        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')

        try:
            basename = os.path.basename(path)
            nwb = nap.load_file(os.path.join(path, "kilosort4", basename + ".nwb"))
            spikes = nwb['units']
            spikes = spikes.getby_threshold("rate", 1)            
        except:
            pass        

        hmm_eps = []
        try:
            filepath = os.path.join(data_directory, s, os.path.basename(s))
            hmm_eps.append(nap.load_file(filepath+"_HMM_ep0.npz"))
            hmm_eps.append(nap.load_file(filepath+"_HMM_ep1.npz"))
            hmm_eps.append(nap.load_file(filepath+"_HMM_ep2.npz"))
        except:
            pass

        idx = spikes._metadata[spikes._metadata["location"].str.contains("adn")].index.values
        spikes = spikes[idx]
        
        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        
        SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
        spikes.set_info(SI)


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
        
        if len(tokeep) > 6:

            spikes = spikes[tokeep]
            # spikes = spikes.getby_threshold('SI', 0.4)
            # groups = spikes._metadata.loc[tokeep].groupby("location").groups
            tcurves         = tuning_curves[tokeep]
            peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

            try:
                velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
            except:
                velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)
            
            ############################################################################################### 
            # PEARSON CORRELATION
            ###############################################################################################
            rates = {}
            for e, ep, bin_size, std in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], [0.1, 0.1, 0.02], [1.5, 1.5, 1.5]):
                ep = ep.drop_short_intervals(bin_size*22)
                count = spikes.count(bin_size, ep)
                rate = count/bin_size
                # rate = rate.as_dataframe()
                rate = rate.smooth(std=bin_size*std, windowsize=bin_size*20).as_dataframe()
                rate = rate.apply(zscore)
                rates[e] = rate
                if len(hmm_eps) and e == "sws":
                    for i in range(len(hmm_eps)):
                        rates["ep"+str(i)] = nap.TsdFrame(rate).restrict(hmm_eps[i])

            
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
            
            #######################
            # COMPUTING PEARSON R FOR EACH SESSION
            #######################
            pearson[s] = np.zeros((6))*np.nan
            pearson[s][0] = scipy.stats.pearsonr(r['wak'], r['rem'])[0]
            pearson[s][1] = scipy.stats.pearsonr(r['wak'], r['sws'])[0]
            if len(hmm_eps):
                for i in range(len(hmm_eps)):
                    pearson[s][i+2] = scipy.stats.pearsonr(r['wak'], r["ep{}".format(i)])[0]

            pearson[s][-1] = len(spikes)

            #######################
            # SAVING
            #######################
            allr.append(r)

allr = pd.concat(allr, 0)

pearson = pd.DataFrame(pearson).T
pearson.columns = ['rem', 'sws', 'ep0', 'ep1', 'ep2', 'count']

datatosave = {
    'allr':allr,
    'pearsonr':pearson
    }

# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'wb'))

# %%

figure()
for i, e in enumerate(["rem", "sws"]):
    subplot(1, 3, i+1)
    tmp = allr[['wak', e]].dropna()
    plot(tmp['wak'], tmp[e], 'o', color = 'red', alpha = 0.5)
    m, b = np.polyfit(tmp['wak'].values, tmp[e].values, 1)
    x = np.linspace(tmp['wak'].min(), tmp['wak'].max(),5)
    plot(x, x*m + b)
    xlabel('wake')
    ylabel(e)
    xlim(-1, 1)
    ylim(-1, 1)
    r, p = scipy.stats.pearsonr(tmp['wak'], tmp[e])
    title('r = '+str(np.round(r, 3)))

subplot(133)
plot(np.zeros(len(pearson))+np.random.randn(len(pearson))*0.1, pearson['rem'].values, 'o') 
plot(np.ones(len(pearson))+np.random.randn(len(pearson))*0.1, pearson['sws'].values, 'o') 


print(scipy.stats.wilcoxon(pearson.dropna()["rem"], pearson.dropna()["sws"]))
print(scipy.stats.ttest_ind(pearson.dropna()["rem"], pearson.dropna()["sws"]))


xticks([0,1], ['rem', 'sws'])
ylim(-0.5, 1)

show()

# %%
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-08-15 13:27:39
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
from GLM_HMM import GLM_HMM_nemos
# from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
import nemos as nmo
import nwbmatic as ntm

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


# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#')

SI_thr = {
    'adn':0.5, 
    'lmn':0.1,
    'psb':1.5
    }

allr = []
allr_glm = []
durations = []
spkcounts = []
corr = []



for s in datasets:

    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    if os.path.isdir(os.path.join(path, "pynapplenwb")):

        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake'].loc[[0]]
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')
        # down_ep = data.read_neuroscope_intervals('down')
        idx = spikes._metadata[spikes._metadata["location"].str.contains("adn")].index.values
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

        spikes = spikes.getby_threshold("SI", SI_thr["adn"])
        spikes = spikes.getby_threshold("rate", 1.0)
        spikes = spikes.getby_threshold("max_fr", 3.0)

        tokeep = spikes.index
        tcurves = tcurves[tokeep]
        peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        order = np.argsort(peaks.values)
        spikes.set_info(order=order, peaks=peaks)

        try:
            maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
            
        except:
            meanwf, maxch = data.load_mean_waveforms(spike_count=100)
            maxch.to_csv(data.nwb_path + "/maxch.csv")        

        spikes.set_info(maxch = maxch[tokeep])

        print(s, tokeep)

        if len(tokeep) > 5:
            
            # figure()
            # for i in range(len(tokeep)):
            #     subplot(4, 4, i+1, projection='polar')
            #     plot(tcurves[tokeep[i]])
            # show()
            
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1).merge_close_intervals(1)

            ############################################################################################### 
            # HMM GLM
            ###############################################################################################
            
            bin_size = 0.02
            window_size = bin_size*50.0


            # GLM
            ############################################
            print("fitting GLM")

            basis = nmo.basis.RaisedCosineBasisLog(
                n_basis_funcs=3, shift=False, mode="conv", window_size=int(window_size/bin_size)
                )

            mask = np.repeat(1-np.eye(len(spikes)), 3, axis=0)


            Y = spikes.count(bin_size, newwake_ep)

            spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(newwake_ep))
            spikes2.set_info(maxch = spikes._metadata["maxch"], group = spikes._metadata["group"])
            Y2 = spikes2.count(bin_size, newwake_ep)

            spikes0 = nap.TsGroup({i:nap.Ts(np.array([])) for i in spikes.keys()}, time_support=newwake_ep)
            Y0 = spikes0.count(bin_size, newwake_ep)

            glm = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"), feature_mask=mask)
            glm.fit(basis.compute_features(Y), Y)

            rglm = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"), feature_mask=mask)
            rglm.fit(basis.compute_features(Y2), Y2)

            glm0 = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"), feature_mask=mask)
            glm0.fit(basis.compute_features(Y0), Y0)

            # HMM
            ############################################
            
            hmm = GLM_HMM_nemos((glm0, glm, rglm))
            # hmm = GLM_HMM((glm, rglm))

            Y = spikes.count(bin_size, sws_ep)
            X = basis.compute_features(Y)
            
            hmm.fit_transition(X, Y)

            ############################################################################################### 
            # GLM CORRELATION
            ############################################################################################### 

            if all([len(ep)>1 for ep in hmm.eps.values()]):
                ############################################################################################### 
                # SAVING HMM EPOCHS
                ###############################################################################################        
                for i in hmm.eps.keys():
                    hmm.eps[i].save(os.path.join(path, os.path.basename(s)+"_HMM_ADN_ep{}".format(i)))

                ############################################################################################### 
                # PEARSON CORRELATION
                ###############################################################################################        
                rates = {}

                for e, ep, bin_size, std in zip(['wak', 'sws'], [newwake_ep, sws_ep], [0.3, 0.03], [1, 1]):
                    count = spikes.count(bin_size, ep)
                    rate = count/bin_size
                    rate = rate.as_dataframe()
                    # rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
                    rate = rate.apply(zscore)                    
                    rates[e] = nap.TsdFrame(rate)
                
                eps = hmm.eps
                for i in eps.keys():
                    rates['ep'+str(i)] = rates['sws'].restrict(eps[i])

                # _ = rates.pop("sws")

                # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
                pairs = [data.basename+"_"+i+"-"+j for i,j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
                r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

                for ep in rates.keys():
                    tmp = np.corrcoef(rates[ep].values.T)
                    if len(tmp):
                        r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

                # Different channels
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
                for i in range(hmm.K):
                    tmp.loc[data.basename,'ep'+str(i)] = scipy.stats.pearsonr(r['wak'], r['ep'+str(i)])[0]
                

                # # flippinge eps
                # # if tmp.iloc[0, 1]>tmp.iloc[0,0]:
                # ido = np.hstack(([0], np.argsort(tmp[['ep1', 'ep2']].values[0])+1))
                # if np.any(ido != np.arange(len(eps))):
                #     r.columns = pd.Index(['wak', 'sws'] + ['ep'+str(i) for i in ido])
                #     r = r[['wak', 'sws']+['ep'+str(i) for i in range(len(eps))]]
                #     tmp.columns = pd.Index(['sws'] + ['ep'+str(i) for i in ido])
                #     tmp = tmp[['sws'] + ['ep'+str(i) for i in range(len(eps))]]
                #     eps = [eps[i] for i in ido]

                corr.append(tmp)
                            
                #######################
                # SAVING
                #######################
                allr.append(r)
                # allr_glm.append(rglm)
                durations.append(pd.DataFrame(data=[e.tot_length('s') for e in eps.values()], columns=[data.basename]).T)
                
                spkcounts.append(
                    pd.DataFrame(data = [[len(spikes.restrict(eps[i]).to_tsd()) for i in range(len(eps))]],
                        columns = np.arange(len(eps)),
                        index = [data.basename])
                    )
            

allr = pd.concat(allr)
# allr_glm = pd.concat(allr_glm, 0)
durations = pd.concat(durations)
corr = pd.concat(corr)
spkcounts = pd.concat(spkcounts)


print(scipy.stats.wilcoxon(corr.iloc[:,0], corr.iloc[:,-2]))
print(scipy.stats.wilcoxon(corr.iloc[:,0], corr.iloc[:,-1]))
print(scipy.stats.wilcoxon(corr.iloc[:,-2], corr.iloc[:,-1]))


figure()
epochs = ['sws'] + ['ep'+str(i) for i in range(len(eps))]
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
tmp = durations.values
tmp = tmp/tmp.sum(1)[:,None]
plot(tmp.T, 'o', color = 'grey')
plot(tmp.mean(0), 'o-', markersize=20)
ylim(0, 1)
title("Durations")

subplot(gs[1,1])
for i, e in enumerate(corr.columns):
    plot(np.random.randn(len(corr))*0.1+np.ones(len(corr))*i, corr[e], 'o')
ylim(0, 1)
xticks(np.arange(corr.shape[1]), corr.columns)

subplot(gs[1,2])
spkcounts = spkcounts.div(spkcounts.sum(axis=1), axis=0)
tmp = spkcounts.values.T
# tmp = tmp/tmp.sum(0)
plot(tmp, 'o-')

show()


##################################################################
# FOR FIGURE 1
##################################################################

datatosave = {
    "corr":corr,
    "allr":allr,
    "durations":durations,
    # "hmm":hmm,
    # "glm":glm,
    # "glmr":rglm
}


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
today = datetime.date.today()
file_name = "GLM_HMM_ADN_"+ today.strftime("%d-%m-%Y") + ".pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)



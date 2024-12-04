# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-03 21:14:15
# %%
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
from GLM_HMM import GLM_HMM_nemos
# from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
import nemos as nmo
import nwbmatic as ntm

# nap.nap_config.set_backend("jax")
nap.nap_config.suppress_conversion_warnings = True

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
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'

# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


allr = {'adn':[], 'lmn':[]}
durations = []
spkcounts = []
corr = {'adn':[], 'lmn':[]}

for s in datasets:
# for s in ['LMN-ADN/A5044/A5044-240401A']:
# for s in ['LMN/A1411/A1411-200910A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

    if os.path.exists(filepath):

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
        sws_ep = nwb['sws']
        rem_ep = nwb['rem']

        # sys.exit()

        spikes = spikes[(spikes.location == "adn") | (spikes.location == "lmn")]
        
        if len(spikes):
            ############################################################################################### 
            # COMPUTING TUNING CURVES
            ###############################################################################################
            tuning_curves = nap.compute_1d_tuning_curves(
                spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                ep = position.time_support.loc[[0]])
            tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
            
            SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
            spikes.set_info(SI)
            
            spikes = spikes[spikes.SI>0.1]

            # CHECKING HALF EPOCHS
            wake2_ep = splitWake(position.time_support.loc[[0]])    
            tokeep2 = []
            stats2 = []
            tcurves2 = []   
            for i in range(2):
                tcurves_half = nap.compute_1d_tuning_curves(
                    spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                    ep = wake2_ep[i]
                    )
                tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

                tokeep, stat = findHDCells(tcurves_half)
                tokeep2.append(tokeep)
                stats2.append(stat)
                tcurves2.append(tcurves_half)       
            tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
            # try:
            #     maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
                
            # except:
            #     meanwf, maxch = data.load_mean_waveforms(spike_count=100)
            #     maxch.to_csv(data.nwb_path + "/maxch.csv")        
            
            spikes = spikes[tokeep]
            tcurves         = tuning_curves[tokeep]
            peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

            # spikes.set_info(maxch = maxch[tokeep])
            if np.sum(spikes.location=="lmn") > 5:
                
                allspikes = spikes
                
                spikes = spikes[spikes.location=='lmn']
                
                # figure()
                # for i in range(len(tokeep)):
                #     subplot(4, 6, i+1, projection='polar')
                #     plot(tcurves[tokeep[i]])
                
                try: 
                    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                    newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
                except:
                    velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                    newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)
                
                ############################################################################################### 
                # HMM GLM
                ###############################################################################################
                
                bin_size = 0.01
                window_size = bin_size*50.0
                

                # GLM
                ############################################
                print("fitting GLM")

                basis = nmo.basis.RaisedCosineBasisLog(
                    n_basis_funcs=3, shift=False, mode="conv", window_size=int(window_size/bin_size)
                    )

                mask = np.repeat(1-np.eye(len(spikes)), 3, axis=0)

                # Ring
                Y = spikes.count(bin_size, newwake_ep)
                glm = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", feature_mask=mask, solver_name="LBFGS")
                glm.fit(basis.compute_features(Y), Y)

                # Random
                spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(newwake_ep))
                spikes2.set_info(group = spikes._metadata["group"])
                Y2 = spikes2.count(bin_size, newwake_ep)

                rglm = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", feature_mask=mask, solver_name="LBFGS")
                rglm.fit(basis.compute_features(Y2), Y2)
                
                # Null
                spikes0 = nap.TsGroup(
                    {i:nap.Ts(
                        np.sort(np.random.choice(spikes[i].t, int(0.0*len(spikes[i])), replace=False))
                        ) for i in spikes.keys()}, time_support=newwake_ep)
                Y0 = spikes0.count(bin_size, newwake_ep)
                
                glm0 = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", feature_mask=mask, solver_name="LBFGS")
                glm0.fit(basis.compute_features(Y0), Y0)


                # HMM
                ############################################
                
                hmm = GLM_HMM_nemos((glm0, glm, rglm))
                # hmm = GLM_HMM((glm, rglm))

                Y = spikes.count(bin_size, sws_ep)
                X = basis.compute_features(Y)
                
                hmm.fit_transition(X, Y)
                

                

                # #ex_ep = nap.IntervalSet(10810.81, 10847.15)
                # #(10846, 10854)

                # figure()
                # gs = GridSpec(4,1)
                # ax = subplot(gs[0,0])
                # plot(hmm.Z)
                # ylabel("state")     
                
                # subplot(gs[1,0], sharex=ax)
                # plot(spikes_adn.restrict(sws_ep).to_tsd("peaks"), '|', markersize=20)

                # subplot(gs[2,0], sharex=ax)
                # plot(spikes.restrict(sws_ep).to_tsd("peaks"), '|', markersize=20)
                # ylabel("Spikes")
                # ylim(0, 2*np.pi)
                
                # subplot(gs[3,0], sharex=ax)
                # [plot(hmm.O[:,i], label = str(i)) for i in range(3)]
                # ylabel("P(O)")
                # legend()


                # show()
                # sys.exit()
                
                
                if all([len(ep)>1 for ep in hmm.eps.values()]):
                    ##############################################################g################################# 
                    # SAVING HMM EPOCHS
                    ###############################################################################################        
                    # for i in hmm.eps.keys():
                    #     hmm.eps[i].save(os.path.join(path, os.path.basename(s)+"_HMM_LMN_ep{}".format(i)))

                    ############################################################################################### 
                    # PEARSON CORRELATION
                    ###############################################################################################                                            
                    rates = {}
                    for e, ep, bin_size, std in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], [0.1, 0.1, 0.02], [3, 3, 3]):
                        ep = ep.drop_short_intervals(bin_size*22)
                        count = allspikes.count(bin_size, ep)
                        rate = count/bin_size
                        # rate = rate.as_dataframe()
                        rate = rate.smooth(std=bin_size*std, windowsize=bin_size*20).as_dataframe()
                        rate = rate.apply(zscore)            
                        rates[e] = nap.TsdFrame(rate)

                    
                    
                    eps = hmm.eps
                    for i in eps.keys():
                        rates['ep'+str(i)] = rates['sws'].restrict(eps[i])

                    structs = ['lmn']
                    if np.sum(allspikes.location=="adn") > 2:
                        structs.append("adn")

                    for k in structs:
                        idx = allspikes.index[allspikes.location==k]
                        pairs = [basename+"_"+i+"-"+j for i,j in list(combinations(idx.astype(str), 2))]

                        r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

                        for ep in rates.keys():
                            tmp = np.corrcoef(rates[ep].as_dataframe()[idx].values.T)
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
                        # Angular differences
                        #######################
                        for p, (i, j) in zip(pairs, list(combinations(idx, 2))):
                            r.loc[p, 'ang'] = min(np.abs(peaks[i] - peaks[j]), 2*np.pi-np.abs(peaks[i] - peaks[j]))
                        
                        #######################
                        # Session correlation
                        #######################
                        if k == "lmn":
                            tmp = pd.DataFrame(index=[basename])
                            tmp['sws'] = scipy.stats.pearsonr(r['wak'], r['sws'])[0]
                            for i in range(hmm.K):
                                tmp.loc[basename,'ep'+str(i)] = scipy.stats.pearsonr(r['wak'], r['ep'+str(i)])[0]
                    
                            corr[k].append(tmp)

                        #######################
                        # SAVING
                        #######################
                            
                        allr[k].append(r)
                        

                    durations.append(pd.DataFrame(data=[e.tot_length('s') for e in eps.values()], columns=[basename]).T)
                    
                    spkcounts.append(
                        pd.DataFrame(data = [[len(spikes.restrict(eps[i]).to_tsd()) for i in range(len(eps))]],
                            columns = np.arange(len(eps)),
                            index = [basename])
                        )
#%%

for k in ['adn', 'lmn']:
    allr[k] = pd.concat(allr[k])

corr["lmn"] = pd.concat(corr["lmn"])

durations = pd.concat(durations)
spkcounts = pd.concat(spkcounts)

# corr = corr.dropna()
# allr = allr.dropna()

# print(scipy.stats.wilcoxon(corr.iloc[:,0], corr.iloc[:,-2]))
# print(scipy.stats.wilcoxon(corr.iloc[:,0], corr.iloc[:,-1]))
# print(scipy.stats.wilcoxon(corr.iloc[:,-2], corr.iloc[:,-1]))

# %%
figure()
epochs = ['sws'] + ['ep'+str(i) for i in range(len(eps))]
gs = GridSpec(2, len(epochs))
for i, e in enumerate(epochs):
    subplot(gs[0,i])
    plot(allr['lmn']['wak'], allr['lmn'][e], 'o', color = 'red', alpha = 0.5)
    m, b = np.polyfit(allr['lmn']['wak'].values, allr['lmn'][e].values, 1)
    x = np.linspace(allr['lmn']['wak'].min(), allr['lmn']['wak'].max(),5)
    plot(x, x*m + b)
    xlabel('wak')
    ylabel(e)
    xlim(allr['lmn']['wak'].min(), allr['lmn']['wak'].max())
    ylim(allr['lmn'].iloc[:,1:-1].min().min(), allr['lmn'].iloc[:,1:-1].max().max())

    r, p = scipy.stats.pearsonr(allr['lmn']['wak'], allr['lmn'][e])
    title('r = '+str(np.round(r, 3)))

subplot(gs[1,0])
tmp = durations.values
tmp = tmp/tmp.sum(1)[:,None]
plot(tmp.T, 'o', color = 'grey')
plot(tmp.mean(0), 'o-', markersize=20)
ylim(0, 1)
title("Durations")

subplot(gs[1,1])
for i, e in enumerate(corr['lmn'].columns):
    plot(np.random.randn(len(corr['lmn']))*0.1+np.ones(len(corr['lmn']))*i, corr['lmn'][e], 'o')
ylim(0, 1)
xticks(np.arange(corr['lmn'].shape[1]), corr['lmn'].columns)

subplot(gs[1,2])
spkcounts = spkcounts.div(spkcounts.sum(axis=1), axis=0)
tmp = spkcounts.values.T
# tmp = tmp/tmp.sum(0)
plot(tmp, 'o-')

show()

# %%


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
file_name = "GLM_HMM_LMN_"+ today.strftime("%d-%m-%Y") + ".pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)



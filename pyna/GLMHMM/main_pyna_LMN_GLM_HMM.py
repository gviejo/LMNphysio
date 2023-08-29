# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   gviejo
# @Last Modified time: 2023-08-28 12:10:22
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
from sklearn.linear_model import PoissonRegressor
from GLM_HMM import GLM_HMM
from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson



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

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


SI_thr = {
    'adn':0.5, 
    'lmn':0.2,
    'psb':1.5
    }

allr = []
allr_glm = []
durations = []
spkcounts = []
corr = []

for s in datasets:
# for s in ["LMN-ADN/A5011/A5011-201010A"]:
# for s in ["LMN/A6701/A6701-201208A"]:
# for s in ['LMN-ADN/A5002/A5002-200303B']:
# for s in ['LMN-PSB/A3010/A3010-210324A']:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    if os.path.isdir(os.path.join(path, "pynapplenwb")):

        data = nap.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake'].loc[[0]]
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

        try:
            maxch = pd.read_csv(data.nwb_path + "/maxch.csv", index_col=0)['0']
            
        except:
            meanwf, maxch = data.load_mean_waveforms(spike_count=100)
            maxch.to_csv(data.nwb_path + "/maxch.csv")        

        spikes.set_info(maxch = maxch[tokeep])

        if len(tokeep) > 5:
            
            # figure()
            # for i in range(len(tokeep)):
            #     subplot(4, 4, i+1, projection='polar')
            #     plot(tcurves[tokeep[i]])
            
            
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.05).time_support.drop_short_intervals(1).merge_close_intervals(1)

            ############################################################################################### 
            # HMM GLM
            ###############################################################################################
            
            bin_size = 0.005
            window_size = bin_size*50.0



            ############################################
            print("fitting GLM")
            glm = ConvolvedGLM(spikes, bin_size, window_size, newwake_ep)
            glm.fit_scipy()
            # glm.fit_sklearn()            

            spikes2 = nap.randomize.shuffle_ts_intervals(spikes.restrict(newwake_ep))            
            spikes2.set_info(maxch = spikes._metadata["maxch"], group = spikes._metadata["group"])
            rglm = ConvolvedGLM(spikes2, bin_size, window_size, newwake_ep)
            rglm.fit_scipy()
            # rglm.fit_sklearn()

            # glm0 = ConvolvedGLM(spikes, bin_size, window_size, newwake_ep)
            # glm0.W = np.zeros_like(glm.W)

            # sys.exit()

            # hmm = GLM_HMM((glm0, glm, rglm))
            hmm = GLM_HMM((glm, rglm))
            
            hmm.fit_transition(spikes, sws_ep, bin_size)


            # figure()
            # gs = GridSpec(3,1)
            # ax = subplot(gs[0,0])
            # plot(hmm.Z)       
            # ylabel("state")     
            # subplot(gs[1,0], sharex=ax)
            # plot(spikes.restrict(sws_ep).to_tsd("peaks"), '|', markersize=20)
            # ylabel("Spikes")
            # ylim(0, 2*np.pi)
            # subplot(gs[2,0], sharex=ax)
            # plot(hmm.time_idx, hmm.O[:,0:])
            # ylabel("P(O)")

            # gs2 = GridSpecFromSubplotSpec(3,1,gs[3,0])
            # sg1 = subplot(gs2[0,0], sharex =ax)
            # sg2 = subplot(gs2[1,0], sharex =ax)
            # sg3 = subplot(gs2[2,0], sharex =ax)
            # # for i in range(1, 3):
            # for i in range(hmm.K):
            #     ep = nap.IntervalSet(start=[1216], end = [1225])
            #     mu = hmm.glms[i].predict(hmm.X)
            #     # mu /= 100.
            #     p = poisson.pmf(k=hmm.Y, mu=mu)
            #     p = p[:,order]
            #     p = nap.TsdFrame(t=hmm.time_idx, d=p)
            #     p = p.restrict(ep)
            #     mu = mu[:,order]
            #     mu = nap.TsdFrame(t=hmm.time_idx, d=mu)
            #     mu = mu.restrict(ep)
            #     sg1.plot(p[0])
            #     sg2.plot(mu[0])
            # y = hmm.Y[:,order]
            # y = nap.TsdFrame(t=hmm.time_idx, d=y)
            # y = y.restrict(ep)
            # sg3.plot(y[0], color='green')

            # sys.exit()
            

            # figure()
            # for i in range(len(spikes)):
            #     w = glm.W[:,i]
            #     w = w[0:-1]
            #     a = peaks.values[list(set(np.arange(len(spikes))) - set([i]))]
            #     tmp = pd.DataFrame(index=a, data=w.reshape(int(w.shape[0]/glm.B.shape[1]), glm.B.shape[1]))
            #     tmp = tmp.sort_index()
            #     subplot(3, 5, i+1)
            #     plot(tmp, 'o-')
            #     plot([peaks.values[i], peaks.values[i]], [0, w.max()])
            # show()            

            # sys.exit()            
            
            ############################################################################################### 
            # GLM CORRELATION
            ############################################################################################### 
            # corrglm = CorrelationGLM(spikes, data.basename)

            # cc_glm = {'wak':corrglm.fit(newwake_ep, 0.3, 3.0)[0]}

            # eps = hmm.eps            
            # for i in range(len(eps)):
            #     cc_glm['ep'+str(i)] = corrglm.fit(eps[i], 0.01, 0.5)[0]

            # rglm = pd.DataFrame(index = cc_glm['wak'].columns, columns = cc_glm.keys())
            # for e in cc_glm.keys():
            #     rglm[e] = cc_glm[e].loc[0]

            if all([len(ep)>1 for ep in hmm.eps.values()]):
                ############################################################################################### 
                # SAVING HMM EPOCHS
                ###############################################################################################        
                for i in hmm.eps.keys():
                    hmm.eps[i].save(os.path.join(path, os.path.basename(s)+"_HMM_ep{}".format(i)))

                ############################################################################################### 
                # PEARSON CORRELATION
                ###############################################################################################        
                rates = {}

                for e, ep, bin_size, std in zip(['wak', 'sws'], [newwake_ep, sws_ep], [0.3, 0.03], [1, 1]):
                    count = spikes.count(bin_size, ep)
                    rate = count/bin_size
                    rate = rate.as_dataframe()
                    rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
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

                to_keep = []
                for p in r.index:
                    tmp = spikes._metadata.loc[np.array(p.split("_")[1].split("-"), dtype=np.int32), ['group', 'maxch']]
                    if tmp['group'].iloc[0] == tmp['group'].iloc[1]:
                        if tmp['maxch'].iloc[0] != tmp['maxch'].iloc[1]:
                            to_keep.append(p)
                r = r.loc[to_keep]
                
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


            

allr = pd.concat(allr, 0)
# allr_glm = pd.concat(allr_glm, 0)
durations = pd.concat(durations, 0)
corr = pd.concat(corr, 0)
spkcounts = pd.concat(spkcounts)

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
    "hmm":hmm,
    "glm":glm,
    "glmr":rglm
}


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
today = datetime.date.today()
file_name = "GLM_HMM_LMN_"+ today.strftime("%d-%m-%Y") + ".pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)
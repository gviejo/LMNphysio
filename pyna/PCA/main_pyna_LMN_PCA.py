# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-10-09 11:43:38
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
# from GLM_HMM import GLM_HMM
# from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap



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
            newwake_ep = np.abs(velocity).threshold(0.02).time_support.drop_short_intervals(1).merge_close_intervals(1)

            ############################################################################################### 
            # HMM GLM
            ###############################################################################################
                        
            bin_size = 0.2
            count = spikes.count(bin_size, newwake_ep)
            tmp = gaussian_filter1d(count.values, 1)
            x = StandardScaler().fit_transform(tmp)
            
            # pca = PCA()
            # y = pca.fit_transform(x)
            # isomap = Isomap(n_components=3, n_neighbors=20)
            # y = isomap.fit_transform(x)
            pca = KernelPCA(kernel="cosine")
            y = pca.fit_transform(x)

            rgb = getRGB(position['ry'], newwake_ep, bin_size)

            beta = np.arccos(np.linalg.norm(y[:,0:2], axis=1)/np.linalg.norm(y, axis=1))
            beta = nap.Tsd(t=count.index.values, d=beta, time_support = newwake_ep)


            figure()
            ax = subplot(211)
            plot(spikes.restrict(newwake_ep).to_tsd("peaks"), '|')
            plot(position['ry'].restrict(newwake_ep), label="Head-direction")
            legend()
            subplot(212, sharex=ax)
            axhline(0)
            axhline(np.pi/2)
            yticks([0, np.pi/2], [0, 90])
            plot(beta)
            ylabel("Angle to plan")
            xlabel("Time (s)")
            show()

            # fig = figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(y[:,0], y[:,1], y[:,2], c=rgb)

            # figure()
            # scatter(y[:,0], y[:,1], c = rgb)
            # show()

            # sys.exit()
            
            count = spikes.count(0.02, sws_ep)
            tmp = gaussian_filter1d(count.values, 5)
            xs = StandardScaler().fit_transform(tmp)

            # ys = isomap.transform(xs)
            ys = pca.transform(xs)


            # COmputing the angular distance
            alpha = np.arccos(np.linalg.norm(ys[:,0:2], axis=1)/np.linalg.norm(ys, axis=1))            
            alpha = nap.Tsd(t=count.index.values, d=alpha, time_support = sws_ep)
                
            figure()
            ax = subplot(211)
            plot(spikes.restrict(sws_ep).to_tsd("peaks"), '|')            
            subplot(212, sharex=ax)
            plot(alpha)
            show()

            sys.exit()

            ep1 = alpha.threshold(np.percentile(alpha, 0.9)).time_support
            ep0 = sws_ep.set_diff(ep1)
            
            eps = {0:ep0, 1:ep1}

            if all([len(ep)>1 for ep in eps.values()]):

                ############################################################################################### 
                # PEARSON CORRELATION
                ###############################################################################################        
                rates = {}

                for e, ep, bin_size, std in zip(['wak', 'sws'], [newwake_ep, sws_ep], [0.2, 0.02], [1, 1]):
                    count = spikes.count(bin_size, ep)
                    rate = count/bin_size
                    rate = rate.as_dataframe()
                    rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
                    rate = rate.apply(zscore)                    
                    rates[e] = nap.TsdFrame(rate)
                
                for i in eps.keys():
                    rates['ep'+str(i)] = rates['sws'].restrict(eps[i])

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
                for i in range(2):
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


# ##################################################################
# # FOR FIGURE 1
# ##################################################################

# datatosave = {
#     "corr":corr,
#     "allr":allr,
#     "durations":durations,
#     "hmm":hmm,
#     "glm":glm,
#     "glmr":rglm
# }


# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# today = datetime.date.today()
# file_name = "GLM_HMM_LMN_"+ today.strftime("%d-%m-%Y") + ".pickle"

# import _pickle as cPickle

# with open(os.path.join(dropbox_path, file_name), "wb") as f:
#     cPickle.dump(datatosave, f)
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-10-07 18:27:21
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
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


SI_thr = {
    'adn':0.5, 
    'lmn':0.2,
    'psb':1.5
    }

allrt = {}

for t in np.arange(20, 100, 5):

    allr = []

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
                tmp = gaussian_filter1d(count.values, 2)
                x = StandardScaler().fit_transform(tmp)
                
                pca = PCA()
                y = pca.fit_transform(x)
                # isomap = Isomap(n_components=3, n_neighbors=20)
                # y = isomap.fit_transform(x)

                rgb = getRGB(position['ry'], newwake_ep, bin_size)

                # fig = figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(y[:,0], y[:,1], y[:,2], c=rgb)

                # figure()
                # scatter(y[:,0], y[:,1], c = rgb)
                # show()

                # sys.exit()
                
                count = spikes.count(0.02, sws_ep)
                tmp = gaussian_filter1d(count.values, 2)
                xs = StandardScaler().fit_transform(tmp)

                # ys = isomap.transform(xs)
                ys = pca.transform(xs)


                # COmputing the angular distance
                alpha = np.arccos(np.linalg.norm(ys[:,0:2], axis=1)/np.linalg.norm(ys, axis=1))            
                alpha = nap.Tsd(t=count.index.values, d=alpha, time_support = sws_ep)
                    
                # figure()
                # plot(alpha)
                # show()

                ep1 = alpha.threshold(np.percentile(alpha, t)).time_support
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

                                
                    #######################
                    # SAVING
                    #######################
                    allr.append(r)

    allr = pd.concat(allr, 0)
    
    r0, _ = scipy.stats.pearsonr(allr['wak'], allr['ep0'])
    r1, _ = scipy.stats.pearsonr(allr['wak'], allr['ep1'])

    allrt[t] = np.array([r0, r1])

df = pd.DataFrame.from_dict(allrt).T

figure()
subplot(121)
plot(df)
subplot(122)
plot(df[1] - df[0])
show()

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-07-07 11:11:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-02-02 15:51:50
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations, product
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from scipy.linalg import hankel
from sklearn.linear_model import PoissonRegressor
from scipy.ndimage import gaussian_filter

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
data_directory = '/mnt/DataRAID2/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')


betamaxt = []
allbetas = []
allmuabetas = []

pairs_info = pd.DataFrame(columns = ['offset', 'session'])

for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')
    down_ep = data.read_neuroscope_intervals('down')
    idx = spikes._metadata[spikes._metadata["location"].str.contains("adn|lmn")].index.values
    spikes = spikes[idx]
      
    ############################################################################################### 
    # COMPUTING TUNING CURVES
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves)    
    tcurves = tuning_curves
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
    peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

    spikes.set_info(SI, peaks=peaks)

    adn = list(spikes.getby_category("location")["adn"].getby_threshold("SI", 0.5).index)
    lmn = list(spikes.getby_category("location")["lmn"].getby_threshold("SI", 0.1).index)

    if len(lmn) > 4 and len(adn) > 4:

        tokeep = adn+lmn
        tokeep = np.array(tokeep)
        spikes = spikes[tokeep]    

        velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
        newwake_ep = velocity.threshold(0.001).time_support 

        tcurves         = tuning_curves[tokeep]
        peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

        ############################################################################################### 
        # GLM
        ###############################################################################################

        pairs = list(product(adn, lmn))

        groups = spikes.getby_category("location")
                

        tref = nap.Ts(
            t=down_ep['start'].values + (down_ep['end'].values - down_ep['start'].values)/2
            )

        for i, p in enumerate(pairs):

            pair_name = data.basename + '_' + str(p[0]) + '_' + str(p[1])

            a = peaks[p[1]] - peaks[p[0]]
            pair_offset = np.abs(np.arctan2(np.sin(a), np.cos(a)))
            pairs_info.loc[pair_name, 'offset'] = pair_offset
            pairs_info.loc[pair_name, 'session'] = s
         
            if pair_offset < (np.pi/4):

                # ## MUA ########
                mua_spikes = {}
                for j, k, gr in zip(range(2), ['adn', 'lmn'], [adn, lmn]):
                    group = list(set(gr) - set([p[j]]))
                    mua_spikes[j] = nap.Ts(t=np.sort(np.hstack([groups[k][j].index.values for j in group])))
                mua_spikes = nap.TsGroup(mua_spikes, time_support = spikes.time_support, location = np.array(['adn', 'lmn']))
                pet2 = nap.compute_perievent(mua_spikes[[1]], tref, minmax=(-0.5, 0.5))
                
                pet = nap.compute_perievent(spikes[list(p)], tref, minmax=(-0.5, 0.5))

                pet_adn = pet[list(pet.keys())[0]].count(0.01)
                pet_lmn = pet[list(pet.keys())[1]].count(0.01)
                pet_mua = pet2[1].count(0.01)

                time_idx = pet_adn.index.values
                
                pet_lmn = StandardScaler().fit_transform(pet_lmn.values.T)
                pet_mua = StandardScaler().fit_transform(pet_mua.values.T)
                pet_adn = pet_adn.values.T

                idx_sort = np.argsort(pet_adn.sum(1))

                pet_lmn = pet_lmn[idx_sort]
                pet_adn = pet_adn[idx_sort]
                pet_mua = pet_mua[idx_sort]

                pet_lmn = pet_lmn[len(pet_lmn)//2:]
                pet_adn = pet_adn[len(pet_adn)//2:]
                pet_mua = pet_mua[len(pet_mua)//2:]

                n = len(time_idx)//4
                idxc = np.arange(n, n*3, 1)

                betas = np.zeros((len(idxc), n*2))
                mbetas = np.zeros((len(idxc), n*2))

                for j, c in enumerate(idxc):
                    reg = np.hstack((pet_lmn[:,c-n:c+n],pet_mua[:,c-n:c+n]))
                    tar = pet_adn[:,c]
                    glm = PoissonRegressor(max_iter = 1000)
                    glm.fit(reg, tar)
                    betas[j] = glm.coef_[0:n*2]
                    mbetas[j] = glm.coef_[n*2:]



                betamax = time_idx[n:n*3][betas.argmax(1)]

                betamaxt.append(
                    pd.DataFrame(
                        index=time_idx[n:n*3], data=betamax, columns=[pair_name]
                        )
                    )
                allbetas.append(betas)
                allmuabetas.append(mbetas)



pairs_info = pairs_info.sort_values(by="offset")

betamaxt = pd.concat(betamaxt, 1)

allbetas = np.array(allbetas)
allmuabetas = np.array(allmuabetas)


figure()

plot(betamaxt.mean(1))


figure()
subplot(121)
imshow(allbetas.mean(0).T)
title("Unit")
n = len(time_idx)//4
idx = np.arange(n, n*3, 5)
xticks(idx-n, time_idx[idx])
yticks(idx-n, time_idx[idx])

subplot(122)
imshow(allmuabetas.mean(0).T)
title("Pop.")
n = len(time_idx)//4
idx = np.arange(n, n*3, 5)
xticks(idx-n, time_idx[idx])
yticks(idx-n, time_idx[idx])



show()
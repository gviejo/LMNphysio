# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 19:20:07
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-17 15:07:15

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
import nemos as nmo
from tqdm import tqdm

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')


scores = {'wak':{}, 'sws':{}}


# for s in datasets:
# for s in ['LMN-ADN/A5030/A5030-220216A']:
for s in ['LMN-ADN/A5024/A5024-210705A']:
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

    if os.path.exists(filepath):
        # sys.exit()
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

        # hmm_eps = []
        # try:
        #     filepath = os.path.join(data_directory, s, os.path.basename(s))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep0.npz"))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep1.npz"))
        #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep2.npz"))
        # except:
        #     pass
        
        spikes = spikes[(spikes.location=="adn")|(spikes.location=="lmn")]
        
        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        
        SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
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
        
        spikes = spikes[tokeep]


        adn = spikes.location[spikes.location=="adn"].index.values
        lmn = spikes.location[spikes.location=="lmn"].index.values


        if len(lmn)>4 and len(adn)>2:
            print(s)
            
            tcurves         = tuning_curves[tokeep]

            try:
                velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
            except:
                velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)


            ############################################################################################### 
            # Population fit
            ###############################################################################################

            for e, ep, bin_size, window_size in zip(['wak', 'sws'], [newwake_ep, sws_ep], [0.01, 0.01], [0.1, 0.1]):

                K = np.eye(int(window_size/bin_size)+1)
                K[1+len(K)//2:,] = 0

                X = spikes[lmn].restrict(ep).count(bin_size).convolve(K)
                X = X[:,:,0:1+len(K)//2]
                X = np.reshape(X, (len(X), -1))
                                
                Y = spikes[adn].restrict(ep).count(bin_size)
            
                glm = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", solver_name="LBFGS")
                glm.fit(X, Y)

                score_1 = glm.score(X, Y, score_type='pseudo-r2-McFadden')

                sys.exit()

                # Random
                # X2 = nap.randomize.resample_timestamps(spikes[lmn].restrict(ep)).count(bin_size, ep).convolve(K)
                # # X2 = nap.randomize.shuffle_ts_intervals(spikes[lmn].restrict(ep)).count(bin_size, ep).convolve(K)
                # X2 = X2[:,:,0:1+len(K)//2]
                # X2 = np.reshape(X2, (len(X2), -1))

                # glm = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", solver_name="LBFGS")
                # glm.fit(X2, Y)

                # score_2 = glm.score(X2, Y, score_type='pseudo-r2-McFadden')


                # scores[e][s] = np.array([score_1, score_2])

                scores[e][s] = np.array([score_1, 0.0])


for e in scores.keys():
    scores[e] = pd.DataFrame.from_dict(scores[e]).T
    scores[e].columns = ['og', 'rnd']

datatosave = {
    'scores':scores
    }
    
dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'SCORES_GLM_LMN-ADN.pickle'), 'wb'))
#    
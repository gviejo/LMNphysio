# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 19:20:07
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-16 14:57:04

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


weights = {}


for s in datasets:
# for s in ['LMN-ADN/A5030/A5030-220216A']:
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

            sys.exit()

            
            Y = spikes.count(bin_size, newwake_ep)
            glm = nmo.glm.PopulationGLM(regularizer_strength=0.001, regularizer="Ridge", feature_mask=mask, solver_name="LBFGS")
            glm.fit(basis.compute_features(Y), Y)



            name = basename    
            pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
            pairs = pd.MultiIndex.from_tuples(pairs)
            
                                            
            for e, ep, bin_size, window_size in zip(['wak', 'sws'], [newwake_ep, sws_ep], 
                [0.1, 0.01], [10, 1]):

                print(e)
                
                count = spikes.restrict(ep).count(bin_size)
                                                
                for p in tqdm(pairs):

                                            
                    n_feature = int(p[0].split("_")[1])
                    n_target = int(p[1].split("_")[1])
                    
                    feat = np.hstack((
                            count.loc[n_feature].convolve(np.eye(int(window_size/bin_size))).values,
                            count.loc[count.columns[count.columns!=n_feature]].sum(1).convolve(np.eye(int(window_size/bin_size))).values
                        ))

                    target = count.loc[n_target]

                    glm = nmo.glm.GLM(regularizer_strength=0.001, regularizer="Ridge", solver_name="LBFGS")

                    glm.fit(feat, target)

                    cc[g][e][p] = glm.coef_[0:len(glm.coef_)//2]
                    cc_mua[g][e][p] = glm.coef_[len(glm.coef_)//2:]

                

# for e in allcc.keys():
#     allcc[e] = pd.concat(allcc[e], axis=1)


# datatosave = {
#     'allcc':allcc,
#     }
    
# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'All_CC_ADN.pickle'), 'wb'))
#    
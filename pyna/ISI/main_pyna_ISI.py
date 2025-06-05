# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-07 10:52:17
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-06-05 17:58:29
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
import sys
sys.path.append("..")
from functions import *

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations


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


datasets = {
    "adn" : np.unique(np.hstack([
        np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
        np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),     
        ])),
    "lmn" : np.unique(np.hstack([
        np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
        np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),     
        ]))
    }


isis = {}
frs = {}

for st in ['adn', 'lmn']:
    ############################################################################################### 
    # GENERAL infos
    ###############################################################################################
    isis[st] = {e:{} for e in ['wak', 'rem', 'sws']}
    frs[st] = {e:[] for e in ['wak', 'rem', 'sws']}

    for s in datasets[st]:
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
            
            
            # hmm_eps = []
            # try:
            #     filepath = os.path.join(data_directory, s, os.path.basename(s))
            #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep0.npz"))
            #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep1.npz"))
            #     hmm_eps.append(nap.load_file(filepath+"_HMM_ep2.npz"))
            # except:
            #     pass

            spikes = spikes[spikes.location == st]

            if len(spikes):
        
                ############################################################################################### 
                # COMPUTING TUNING CURVES
                ###############################################################################################
                tuning_curves = nap.compute_1d_tuning_curves(
                    spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                    ep = position.time_support.loc[[0]])
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
                

                if len(tokeep) > 5 and rem_ep.tot_length('s') > 60:
                    print(s)

                    spikes = spikes[tokeep]
                    # groups = spikes._metadata.loc[tokeep].groupby("location").groups
                    tcurves         = tuning_curves[tokeep]

                    try:
                        velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                        newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
                    except:
                        velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                        newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)
                    
                
                    
                    ############################################################################################### 
                    # ISI
                    ###############################################################################################
                    
                    
                    for e, ep in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep]):
                        isi = {}
                        fr = spikes.restrict(ep).rate
                        fr.index = pd.Index([basename+'_'+str(n) for n in fr.index])
                        frs[st][e].append(fr)
                        for n in spikes.keys():
                            tmp = []
                            for j in ep.index:
                                spk = spikes[n].get(ep.start[j], ep.end[j]).index.values
                                if len(spk)>2:
                                    tmp.append(np.diff(spk))
                            tmp = np.hstack(tmp)
                                                        
                            isis[st][e][basename+'_'+str(n)] = tmp
            
            
for st in frs.keys():
    for e in frs[st].keys():
        frs[st][e] = pd.concat(frs[st][e])

datatosave = {'isis':isis, 'frs':frs}

dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'All_ISI.pickle'), 'wb'))


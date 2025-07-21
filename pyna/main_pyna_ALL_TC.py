# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-21 17:12:04

# %%
import numpy as np
import pandas as pd
import pynapple as nap
# import nwbmatic as ntm
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.stats import zscore

# %%
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

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)


alltc = []
allinfo = []

for s in datasets:

    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    
    if os.path.exists(filepath):
        
        nwb = nap.load_file(filepath)
        
        spikes = nwb['units']
        # spikes = spikes.getby_threshold("rate", 1)

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
        


        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(
            spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
            ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

        tuning_curves.columns = [basename+"_"+str(n) for n in tuning_curves.columns]

        # Mutual Information        
        SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))

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
        
        # Saving
        alltc.append(tuning_curves)

        info = pd.DataFrame(
            index = tuning_curves.columns,
            columns = ['location', 'rate', 'group', 'SI', 'halfr']
            )

        info['location'] = spikes.location.values
        info['rate'] = spikes.rate.values
        info['group'] = spikes.group.values
        info['SI'] = SI.values
        info['tokeep'] = 0
        info.loc[info.index[tokeep],'tokeep'] = 1

        allinfo.append(info)        


allinfo = pd.concat(allinfo, axis=0)

alltc = pd.concat(alltc, axis=1)

tc = centerTuningCurves_with_peak(alltc)

tokeep = allinfo.index.values[allinfo['tokeep']==1]
tc = tc/tc.max()

datatosave = {
    'alltc':alltc,
    'allinfo':allinfo
    }
    
dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'All_TC.pickle'), 'wb'))



figure()
subplot(131)
loglog(allinfo['rate'], allinfo['SI'], '.')
loglog(allinfo['rate'][allinfo['tokeep']==1], allinfo['SI'][allinfo['tokeep']==1], '.')
axvline(1.0)
axhline(0.1)

subplot(132)
imshow(tc[tokeep].values.T, aspect='auto')


show()
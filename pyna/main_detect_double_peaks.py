# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-09 16:24:21

# %%
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
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)


alltc = []


for s in datasets:
# for s in ['LMN-ADN/A5021/A5021-210521A']:
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    print(s)

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
        

        spikes = spikes[(spikes.location == "lmn")|(spikes.location == "adn")]
    
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

            tuning_curves = tuning_curves[spikes.keys()]

            tuning_curves.columns = pd.Index([basename + "_" + str(n) for n in tuning_curves.columns])

            alltc.append(tuning_curves)


alltc = pd.concat(alltc, axis=1)

alltc = centerTuningCurves(alltc)

tmp = np.vstack((alltc.values, alltc.values, alltc.values))

ft = np.abs(np.fft.fft(tmp, axis=0))

freq = np.fft.fftfreq(360, 1/120)

index = np.argsort(ft[freq==2.0][0])

index = np.argsort(ft[(freq>0.5)&(freq<1.5)].mean(0))

newtc = alltc.iloc[:,index]

weights = pd.DataFrame.from_dict(
	{"s":[s.split("_")[0] for s in newtc.columns],
	"w":np.arange(0, len(index))})

meanw = pd.Series({s:weights.loc[g,'w'].mean() for s, g in weights.groupby("s").groups.items()})

order = meanw.index[np.argsort(meanw.values)]




figure()
tmp = newtc.iloc[:,newtc.columns.str.contains(order[1])]
for i in range(tmp.shape[1]):
	subplot(5,5,i+1)
	plot(tmp.iloc[:,i])
show()


figure()
for i in range(100):
	subplot(10,10,i+1, projection='polar')
	plot(newtc.iloc[:,i])

figure()
for i in range(100):
	subplot(10,10,i+1, projection='polar')
	plot(newtc.iloc[:,-i])
show()


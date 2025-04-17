# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-01-04 06:11:33
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-04-17 14:54:15
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore


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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')


allcc = {'wak':[], 'rem':[], 'sws':[]}
allccdown = {'psb':[], 'lmn':[]}

angdiff = {}

hd_info = {}

alltcurves = []

for s in datasets:

    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    # filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    filepath = os.path.join(path, "pynapplenwb", basename + ".nwb")

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
        
        psb_spikes = spikes[spikes.location=="psb"]

        spikes = spikes[(spikes.location=="psb")|(spikes.location=="lmn")]
        
        
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


        psb = spikes.location[spikes.location=="psb"].index.values
        lmn = spikes.location[spikes.location=="lmn"].index.values

    
        print(s)
        
        tcurves         = tuning_curves[tokeep]
        # tcurves = tuning_curves
        


        try:
            velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
        except:
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)


        ############################################################################################### 
        # CROSS CORRELOGRAM
        ###############################################################################################         
        
        
        for e, ep, bin_size, window_size in zip(['wak', 'rem', 'sws'], 
            [newwake_ep, rem_ep, sws_ep], 
            [0.01, 0.01, 0.001], [1, 1, 1]):

            tmp = nap.compute_crosscorrelogram(
                    # tuple(spikes.getby_category("location").values()),
                    (spikes[lmn], spikes[psb]),
                    bin_size, 
                    window_size, 
                    ep, norm=True)        


            pairs = [(basename + "_" + str(n), basename + "_" + str(m)) for n, m in tmp.columns]
            pairs = pd.MultiIndex.from_tuples(pairs)
            tmp.columns = pairs
            
            allcc[e].append(tmp)


        #######################
        # Angular differences
        #######################
        peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        for p in pairs:
            i = int(p[0].split("_")[1])
            j = int(p[1].split("_")[1])
            angdiff[p] = min(np.abs(peaks[i] - peaks[j]), 2*np.pi-np.abs(peaks[i] - peaks[j]))
            
            hd_info[p] = SI.loc[[i,j]].values.flatten()


        tcurves.columns = [basename + "_" + str(n) for n in tcurves.columns]
        alltcurves.append(tcurves)

        # sys.exit()


for e in allcc.keys():
    allcc[e] = pd.concat(allcc[e], axis=1)

angdiff = pd.Series(angdiff)
hd_info = pd.DataFrame(hd_info).T
hd_info.columns = ['lmn', 'psb']

alltcurves = pd.concat(alltcurves, axis=1)


# datatosave = {'allcc':allcc, 'angdiff':angdiff}#, 'allcup': allccdown}
# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'CC_LMN-PSB.pickle'), 'wb'))


figure()
for i, e in enumerate(allcc.keys()):
    subplot(1,3,i+1)
    #plot(allcc[e], alpha = 0.7, color = 'grey')
    plot(allcc[e].mean(1), '.-')
    title(e)


# figure()
# for i,k in enumerate(['psb', 'lmn']):
#     subplot(2,1,i+1)
#     plot(allccdown[k].mean(1))
# show()


angbins = np.linspace(0, np.pi, 4)
idx = np.digitize(angdiff, angbins)-1

ang0 = angdiff[idx==0].index

cc = allcc['sws'][ang0]

cc = cc[cc.idxmax().sort_values().index] 

# imshow(cc.values.T, aspect='auto')

# bins = np.linspace(hd_info['psb'].min(), hd_info['psb'].max(), 9)
bins = np.geomspace(hd_info['psb'].min(), hd_info['psb'].max(), 9)

idx = np.digitize(hd_info['psb'], bins)-1

ccg = {}
for i in np.unique(idx):
    ccg[i] = allcc['sws'][hd_info.index[idx==i]].mean(1)
ccg = pd.DataFrame(ccg)



figure()
for i in range(ccg.shape[1]):
    subplot(3,3,i+1)
    plot(ccg[i].loc[-0.1:0.1])



maxt = allcc['sws'].idxmax()

idx = maxt[(maxt>-0.02) & (maxt<0.0)].index.values

new_idx = allcc['sws'][idx].max().sort_values().index.values[::-1]


figure()
for i in range(54):
    ax = subplot(6,9,i+1)    
    plot(allcc['sws'][new_idx[i]].loc[-0.1:0.1])
    axvline(0)

figure()
for i in range(54):
    ax = subplot(6,9,i+1, projection='polar')    
    p = new_idx[i]
    plot(alltcurves[p[0]])
    plot(alltcurves[p[1]])



figure()
subplot(121, projection='polar')
plot(alltcurves[new_idx[0][0]], label='lmn')
plot(alltcurves[new_idx[0][1]], label='psb')
legend()
subplot(122)
plot(allcc['sws'][new_idx[0]])
show()

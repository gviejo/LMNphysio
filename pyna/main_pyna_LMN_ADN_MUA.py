#!/usr/bin/env python
'''

'''
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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')


allcc = {'wak':[], 'rem':[], 'sws':[]}
allccdown = {'adn':[], 'lmn':[]}

for s in datasets:
# for s in ['LMN-ADN/A5030/A5030-220216A']:
# for s in ['LMN-ADN/A5024/A5024-210705A']:
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
            # MUA CROSS CORRELOGRAM
            ############################################################################################### 

            mua = nap.TsGroup({
                0:spikes[spikes.location=='lmn'].to_tsd(),
                1:spikes[spikes.location=='adn'].to_tsd()
                }, metadata={"location":['lmn', 'adn']}, time_support = spikes.time_support)
            
            for e, ep, bin_size, window_size in zip(
                ['wak', 'rem', 'sws'], 
                [newwake_ep, rem_ep, sws_ep], 
                [0.0005, 0.01, 0.0005], 
                [1, 1, 1]):
                allcc[e].append(
                    nap.compute_crosscorrelogram(mua, bin_size, window_size, ep, norm=True)[(0,1)]
                    )

            # ######################################################################################################
            # # Cross-correlation DOWN center
            # ######################################################################################################    
            # # TAKING UP_EP AND DOWN_EP LARGER THAN 100 ms
            # down_ep = down_ep.drop_short_intervals(50, time_units = 'ms')  

            # tref = nap.Ts(
            #     t=down_ep['start'].values + (down_ep['end'].values - down_ep['start'].values)/2
            #     )

            # for k in ['adn', 'lmn']:
            #     cc_down = nap.compute_eventcorrelogram(groups[k], tref, 0.01, 0.2, sws_ep)

            #     cc_down.columns = [data.basename+'_'+str(n) for n in groups[k].keys()]

            #     allccdown[k].append(cc_down)


# for k in ['adn', 'lmn']:
#     allccdown[k] = pd.concat(allccdown[k], axis=1)

for e in allcc.keys():
    allcc[e] = pd.concat(allcc[e], axis=1)
    allcc[e] = allcc[e].apply(zscore)

datatosave = {'allcc':allcc}#, 'allcup': allccdown}
dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'MUA_LMN-ADN.pickle'), 'wb'))


figure()
for i, e in enumerate(allcc.keys()):
    subplot(1,3,i+1)
    #plot(allcc[e], alpha = 0.7, color = 'grey')
    plot(allcc[e].mean(1), '.-')
show()

# figure()
# for i,k in enumerate(['adn', 'lmn']):
#     subplot(2,1,i+1)
#     plot(allccdown[k].mean(1))
# show()

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

angdiff = {}

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

        nwb.close()
        
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

    
        print(s)
        
        tcurves         = tuning_curves[tokeep]

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
            [0.0005, 0.0005, 0.0005], [0.1, 0.1, 0.1]):

            dict_spk = spikes.getby_category("location")

            tmp = nap.compute_crosscorrelogram(
                    (dict_spk['lmn'], dict_spk['adn']),
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
                # angdiff[p] = min(np.abs(peaks[i] - peaks[j]), 2*np.pi-np.abs(peaks[i] - peaks[j]))

                angdiff[p] = (peaks[i] - peaks[j] + np.pi)%(2*np.pi) - np.pi


for e in allcc.keys():
    allcc[e] = pd.concat(allcc[e], axis=1)

angdiff = pd.Series(angdiff)

# Detecting synaptic connections
def get_zscore(cc, w=0.02):
    # Smoothing with big gaussian
    sigma = int(w/np.median(np.diff(cc.index.values)))+1
    df = cc.apply(lambda col: gaussian_filter1d(col, sigma=sigma))

    # Zscoring
    tmp = cc - df
    zcc = tmp/tmp.std(0)
    return zcc

# Detecting
thr = 3.0
zcc = get_zscore(allcc['sws'])
tmp2 = (zcc>=thr).loc[0.001:0.008]
pc = tmp2.columns[np.any(tmp2 & tmp2.shift(1, fill_value=False), 0)]
zorder = zcc[pc].loc[0.001:0.008].max().sort_values()
order = zorder[::-1].index.values

# Counter 
tmp3 = (zcc>=thr).loc[-0.008:-0.001]
pc = tmp3.columns[np.any(tmp3 & tmp3.shift(1, fill_value=False), 0)]
zcounter = zcc[pc].loc[-0.008:-0.001].max().sort_values()
counter = zcounter[::-1].index.values





figure()
for i, e in enumerate(allcc.keys()):
    subplot(1,3,i+1)
    #plot(allcc[e], alpha = 0.7, color = 'grey')
    plot(allcc[e].mean(1), '.-')



figure()
subplot(121)

zcc_sws = get_zscore(allcc['sws'])
zcc_wak = get_zscore(allcc['wak'])

plot(zcc_sws[angdiff[zorder.index.values].sort_values().index].mean(1), label='sws')
plot(zcc_wak[angdiff[zorder.index.values].sort_values().index].mean(1), label='wak')

subplot(122)

a = zcc_sws[angdiff[zorder.index.values].sort_values().index]
imshow(a.values.T, cmap='turbo', vmax=3)

show()



# figure()
# for i,k in enumerate(['adn', 'lmn']):
#     subplot(2,1,i+1)
#     plot(allccdown[k].mean(1))
# show()


datatosave = {
    'allcc':allcc, 
    'angdiff':angdiff,
    'zcc':{"wak":get_zscore(allcc['wak']),"sws":get_zscore(allcc['sws'])},
    'zorder':zorder,
    'order':order
    }#, 'allcup': allccdown}

dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'CC_LMN-ADN.pickle'), 'wb'))




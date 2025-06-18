# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-06-16 18:35:26

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
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)


allr = []
pearson = {}

for s in datasets:
# for s in ["LMN-ADN/A5011/A5011-201014A"]:
# for s in ['LMN-ADN/A5021/A5021-210521A']:
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

        spikes = spikes[spikes.location == "lmn"]
    
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
            

            if len(tokeep) > 7:
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
                # PEARSON CORRELATION
                ###############################################################################################
                
                # Wake rate                
                rates = {}
                rates2 = {}
                for e, ep, bin_size, std in zip(['wak', 'sws'], [newwake_ep, sws_ep], [0.1, 0.02], [3, 3]):
                    ep = ep.drop_short_intervals(bin_size*22)
                    count = spikes.count(bin_size, ep)
                    rate = count/bin_size
                    # rate = rate.as_dataframe()
                    rate = rate.smooth(std=bin_size*std, windowsize=bin_size*20).as_dataframe()
                    rates2[e] = rate
                    rate = rate.apply(zscore)            
                    rates[e] = rate 
                                
                # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
                pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))
                pairs = pd.MultiIndex.from_tuples(pairs)
                r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

                for ep in rates.keys():
                    tmp = np.corrcoef(rates[ep].values.T)
                    r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]

                name = basename    
                pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
                pairs = pd.MultiIndex.from_tuples(pairs)
                r.index = pairs

                ############################################################################################### 
                # SIGMOIDE
                ###############################################################################################

                def sigmoide(x, beta):
                    return x.mean()/(1+np.exp(-beta*(x-x.mean()/2)))

                def morph_sigmoid_to_linear(x, alpha=0.0):
                    """
                    alpha=0.0 -> sigmoid
                    alpha=1.0 -> linear
                    """
                    s = sigmoide(x, 0.1)
                    # l = (x - x.min()) / (x.max() - x.min())  # Normalize linear to [0,1]
                    l = x
                    return (1 - alpha) * s + alpha * l


                def linear_to_downward_curve(x, curviness=0.0):
                    """
                    Morphs from linear to a downward-bending curve, with fixed endpoints at (0, 0) and (1, 1).

                    Parameters:
                    - x: np.ndarray or float in [0, 1]
                    - curviness: float in [0, 1]; 0 = linear, 1 = fully curved (max downward bend)

                    Returns:
                    - y: same shape as x, values in [0, 1]
                    """
                    max_v = np.nanmean(x)                    
                    # x = np.clip(x, 0, 1)                    
                    
                    # Map curviness [0, 1] to gamma [1, high]; gamma=1 is linear, gamma>1 is concave
                    gamma = 1 + 8 * curviness  # Adjust 4 for more or less curvature at max

                    curved = (x/np.max(x)) ** gamma  # always goes through (0,0) and (1,1)

                    return curved*max_v

                def cut_percentile(x, alpha):
                    y = x.copy()
                    y[y<np.percentile(y, alpha)] = 0.0
                    return y
                
                beta_values = np.linspace(0.001, 5, 20)
                alpha_values = np.linspace(0, 100, 10)[0:-1]
                alpha_values = np.geomspace(0.2, 100, 100)[0:-1]
                alpha_values = (100-np.geomspace(0.1, 100, 50))[::-1]
                
                r_sig = pd.DataFrame(
                    index=pairs, columns=alpha_values
                )
                
                pearson[s] = pd.Series(index=alpha_values)
                
                
                for alpha in alpha_values:
                    new_rate = []
                    for n in rates2['sws'].columns:
                        # new_rate.append(morph_sigmoid_to_linear(rates2['sws'][n].values, alpha))
                        new_rate.append(cut_percentile(rates2['sws'][n].values, alpha))
                    new_rate = np.stack(new_rate).T
                    # tmp = pd.DataFrame(new_rate).apply(zscore)
                    tmp = pd.DataFrame(new_rate)#.apply(zscore)
                    tmp2 = np.corrcoef(tmp.values.T)
                    r_sig[alpha] = tmp2[np.triu_indices(tmp2.shape[0], 1)]

                    pearson[s][alpha] = scipy.stats.pearsonr(r['wak'], r_sig[alpha])[0]
                
                

# allr = pd.concat(allr)

pearson = pd.DataFrame(pearson).T
# pearson.columns = ['rem', 'sws', 'ep0', 'ep1', 'ep2', 'count']

# datatosave = {
#     'allr':allr,
#     'pearsonr':pearson
#     }
    
dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'wb'))


# %%
# print(pearson)
figure()
plot(pearson.mean(0))
# ylim(0, 1)
show()


# %%


# print(pearson)
figure()
plot(pearson.T, '.-')
plot(pearson.mean(0), 'o-')
show()
# %%

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-07 10:52:17
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-02 17:48:50
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
pr2 = {}

for st in ['adn', 'lmn']:
    ############################################################################################### 
    # GENERAL infos
    ###############################################################################################
    isis[st] = {e:{} for e in ['wak', 'rem', 'sws']}
    frs[st] = {e:[] for e in ['wak', 'rem', 'sws']}
    pr2[st] = {e:{} for e in ['wak', 'rem', 'sws']}

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
            
            sys.exit()
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

                if st == "adn":
                    spikes = spikes[spikes.SI>0.3]
                else:
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
                

                if len(tokeep) > 2 and rem_ep.tot_length('s') > 60:
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

                    for n in spikes.keys():
                        
                        isi = nap.Tsd(spikes[n].t[0:-1], np.diff(spikes[n].t))

                        scores = {}

                        for e, ep in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep]):                        

                            tmp = isi.restrict(ep).values

                            ll = evaluate_gmm(np.log(tmp[(tmp>0.002)&(tmp<10.0)]))
                            # scores[e] = 1 - ll[1] / ll[0]

                            # fr = spikes[n].restrict(ep).rate
                            # fr.index = pd.Index([basename+'_'+str(n) for n in fr.index])
                            # frs[st][e].append(fr)
                            
                                
                            
                            pr2[st][e][basename+'_'+str(n)] = 1 - ll[1] / ll[0]



for st in pr2.keys():
    pr2[st] = pd.DataFrame(pr2[st])            
            
# for st in frs.keys():
#     for e in frs[st].keys():
#         frs[st][e] = pd.concat(frs[st][e])

figure()

violinplot(pr2['adn'][['wak', 'sws']], [1,2], showextrema=False, showmeans=True)
violinplot(pr2['lmn'][['wak', 'sws']], [3,4], showextrema=False, showmeans=True)

xticks(np.arange(1, 5), [st+"-"+e for st in ['adn', 'lmn'] for e in ['wak', 'sws']])


ylabel("pseudo-R2")
ylim(-0.05, 0.3)

tight_layout()
savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_isi/"+"isi_gaussian_mixture.pdf"))
show()


datatosave = {'isis':isis, 'frs':frs, 'pr2':pr2}

dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'All_ISI.pickle'), 'wb'))



import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Example data (could be any 1D np.ndarray)
data = np.concatenate([
    np.random.normal(0, 1, 500),
    # np.random.normal(5, 1, 500)
])

X = data.reshape(-1, 1)

# Fit GMMs
gmms = []
for n_components in [1, 2]:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X)
    gmms.append(gmm)

# Plot histogram
plt.figure(figsize=(10, 5))
counts, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.5, color='gray', label='Data histogram')

# Evaluate PDFs over a smooth range
x_plot = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)

colors = ['red', 'blue']
for i, gmm in enumerate(gmms):
    logprob = gmm.score_samples(x_plot)
    pdf = np.exp(logprob)
    label = f'{gmm.n_components} Gaussian{"s" if gmm.n_components > 1 else ""}'
    plt.plot(x_plot, pdf, color=colors[i], label=label, linewidth=2)

plt.legend()
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('GMM fit with 1 vs 2 components')
plt.grid(True)
plt.show()
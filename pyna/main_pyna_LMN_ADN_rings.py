# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 16:35:14
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-11-29 17:01:26
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from functions import *
import sys
from itertools import combinations, product
# from umap import UMAP
from matplotlib.pyplot import *
from sklearn.manifold import Isomap

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

path = os.path.join(data_directory, 'LMN-ADN/A5043/A5043-230306A')
basename = os.path.basename(path)
data = nap.load_file(path + "/kilosort4/" + basename+".nwb")

spikes = data['units']
angle = data['ry']
epochs = data['epochs']
wake_ep = epochs[epochs.tags=="wake"]
sleep_ep = epochs[epochs.tags=="sleep"]
sws_ep = data['sws']

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)

SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)

spikes = spikes[spikes.SI>0.1]

# CHECKING HALF EPOCHS
wake2_ep = splitWake(angle.time_support.loc[[0]])    
tokeep2 = []
stats2 = []
tcurves2 = []   
for i in range(2):
    tcurves_half = nap.compute_1d_tuning_curves(
        spikes, angle, 120, minmax=(0, 2*np.pi), 
        ep = wake2_ep[i]
        )
    tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

    tokeep, stat = findHDCells(tcurves_half)
    tokeep2.append(tokeep)
    stats2.append(stat)
    tcurves2.append(tcurves_half)       
tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

spikes = spikes[tokeep]

groups = spikes.getby_category('location')

# bin_size = 0.2

# umaps = {}
# for g in ['adn', 'lmn']:
# 	count = groups[g].count(bin_size, angle.time_support.loc[[0]])
# 	count = count.as_dataframe()
# 	rate = np.sqrt(count/bin_size)
# 	rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=2)
# 	ump = Isomap(n_neighbors = 40, n_components = 3).fit_transform(rate)
# 	umaps[g] = ump

# rgb = getRGB(angle, angle.time_support.loc[[0]], bin_size)

# figure()
# subplot(121)
# scatter(umaps['adn'][:,0], umaps['adn'][:,1], color=rgb)
# subplot(122)
# scatter(umaps['lmn'][:,0], umaps['lmn'][:,1], c=rgb)
# show()


from sklearn.decomposition import PCA

pca_adn = PCA().fit_transform(tuning_curves[groups['adn'].keys()])
pca_lmn = PCA().fit_transform(tuning_curves[groups['lmn'].keys()])

new_sws_ep = groups['adn'].to_tsd().count(0.02, sws_ep).smooth(0.04).threshold(1.0).time_support

decoded, p = nap.decode_1d(tuning_curves[groups['adn']], groups['adn'], sws_ep, bin_size=0.03, feature=angle)

tc_sws = nap.compute_1d_tuning_curves(groups['adn'], decoded, 120, minmax=(0, 2*np.pi), ep = new_sws_ep)

figure()
plot(tc_sws)

figure()
plot(pca_adn[:,0], pca_adn[:,1], label="ADN")
plot(pca_lmn[:,0], pca_lmn[:,1], label="LMN")
legend()
show()

# ############################################################################################### 
# # FIGURES
# ###############################################################################################

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = spikes._metadata['group']

figure()
count = 1
for j in np.unique(shank):
	neurons = shank.index[np.where(shank == j)[0]]
	for k,n in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		plot(tuning_curves[n], label = str(shank.loc[n]) + ' ' + str(n), color = colors[shank.loc[n]-1])
		legend()
		count+=1
		gca().set_xticklabels([])
show()

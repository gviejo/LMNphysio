# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 16:35:14
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-01 17:36:22
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
from functions import *
import sys
from itertools import combinations, product
from umap import UMAP
from matplotlib.pyplot import *

path = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201014A'
data = nap.load_session(path, 'neurosuite')

spikes = data.spikes.getby_threshold('freq', 1.0)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)

groups = spikes.getby_category('location')

bin_size = 0.2

umaps = {}
for g in ['adn', 'lmn']:
	count = groups[g].count(bin_size, angle.time_support.loc[[0]])
	count = count.as_dataframe()
	rate = np.sqrt(count/bin_size)
	rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
	ump = UMAP(n_neighbors = 40, n_components = 3).fit_transform(rate)
	umaps[g] = ump

rgb = getRGB(angle, angle.time_support.loc[[0]], bin_size)

figure()
subplot(121)
scatter(umaps['adn'][:,0], umaps['adn'][:,1], color=rgb.values)
subplot(122)
scatter(umaps['lmn'][:,0], umaps['lmn'][:,1], c=rgb.values)
show()

# # COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)

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

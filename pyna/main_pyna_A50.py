# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-04-13 09:53:18
# @Last Modified by:   gviejo
# @Last Modified time: 2023-10-29 16:52:47
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
from scipy.ndimage import gaussian_filter1d

# path = '/mnt/DataRAID2/LMN-ADN/A5043/A5043-230306A'
# path = '/mnt/ceph/users/gviejo/LMN-ADN/A5043/A5043-230301A'
path = "/mnt/Data/A5000/A5043-230315A"

data = ntm.load_session(path, 'neurosuite')

# spikes = data.spikes.getby_threshold('rate', 1.0)
spikes = data.spikes
angle = data.position['ry']
wake_ep = data.epochs['wake']


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 60, minmax=(0, 2*np.pi), ep = angle.time_support)
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 1.0)
# SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
# spikes.set_info(SI)

# pf, bins = nap.compute_2d_tuning_curves(spikes, data.position[['x', 'z']], 15, ep=wake_ep)

############################################################################################### 
# PLOT
###############################################################################################
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = spikes._metadata.group.values

figure()
count = 1
for l,j in enumerate(np.unique(shank)):
	neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
	for k,i in enumerate(neurons):		
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')	
		plot(tuning_curves[i], label = str(j+1) + ' ' + str(i), color = colors[l])
		legend()
		count+=1
		gca().set_xticklabels([])

show()


figure()
count = 1
for l,j in enumerate(np.unique(shank)):
	neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
	for k,i in enumerate(neurons):		
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count)	
		imshow(pf[i], aspect = 'auto')
		title(str(shank[l]) + ' ' + str(i))
		legend()
		count+=1
		gca().set_xticklabels([])

show()



ep = nap.IntervalSet(start = data.position.time_support.start[0],
						end = data.position.time_support.start[0]+15*60)

X = np.sqrt(spikes.getby_category("location")['adn'].restrict(ep).count(0.1))
t = X.t
X = X.values

X = X - X.mean(0)
X = X / X.std(0)

from sklearn.manifold import Isomap


imap = Isomap(n_components=2).fit_transform(X)

scatter(imap[:,0], imap[:,1])


alpha = np.arctan2(imap[:,1], imap[:,0])
alpha += np.pi

alpha = nap.Tsd(t=t, d=alpha)

angle2 = angle.restrict(ep).bin_average(0.1)

# alpha -= np.mean(angle2[0:10]) - np.mean(alpha[0:10])
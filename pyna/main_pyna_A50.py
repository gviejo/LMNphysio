# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-04-13 09:53:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-11-03 17:56:27
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

if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'

path = os.path.join(data_directory, 'LMN-ADN/A5043/A5043-230315A')

data = ntm.load_session(path, 'neurosuite')

# spikes = data.spikes.getby_threshold('rate', 1.0)
spikes = data.spikes
angle = data.position['ry']
wake_ep = data.epochs['wake']


# # Angular threhsold
# rz = data.position['rx']
# tmp = (rz>np.pi).values
# rz[tmp] = (rz[tmp] - 2*np.pi).values



# ep = rz.threshold(0.8, "above").threshold(1.4, "below").time_support

# Linear velocity
position = data.position[['x', 'z']]
pos2 = position.bin_average(0.4)
pos2 = pos2.as_dataframe().rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
speed = np.sqrt(np.sum(np.power(pos2.values[1:, :] - pos2.values[0:-1, :], 2), 1))    
speed = nap.Tsd(t = pos2.index.values[0:-1], d=speed, time_support = position.time_support)

ep = speed.threshold(0.006, "above").time_support

# ep = ep1.intersect(ep2)

angle2 = angle.restrict(ep)
# angle2 = smoothAngle(angle, 1)

tuning_curves = nap.compute_1d_tuning_curves(spikes, angle2, 360, minmax=(0, 2*np.pi), ep = angle2.time_support)
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 40, deviation = 3.0)
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


# sys.exit()

# csv_file = os.path.join(path, "A5043-230315A_1_test2.csv")

# position = pd.read_csv(csv_file, header=[3, 4, 5], index_col=1)
# if 1 in position.columns:
#     position = position.drop(labels=1, axis=1)
# position = position[~position.index.duplicated(keep="first")]

# m1 = position['C3510']['Position'][['X', 'Z']].values
# m2 = position['C3520']['Position'][['X', 'Z']].values
# m3 = position['C3530']['Position'][['X', 'Z']].values

# alpha = np.arctan2((m3-m1)[:,1], (m3-m1)[:,0])

# alpha = nap.Tsd(t=angle.t, d=alpha)

# ep = alpha[~np.isnan(alpha).values].find_support(1)

# alpha = alpha.restrict(ep)

# alpha = smoothAngle(alpha, 1)

# tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = ep)
# tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 1.0)





ep2 = nap.IntervalSet(start = data.position.time_support.start[0],
						end = data.position.time_support.start[0]+15*60)

X = np.sqrt(spikes.getby_category("location")['adn'].restrict(ep2).count(0.1))
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
alpha = alpha.bin_average(0.2)


tc2 = nap.compute_1d_tuning_curves(spikes, alpha, 120, minmax=(0, 2*np.pi), ep = ep2)
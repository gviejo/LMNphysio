import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys, os
from sklearn.manifold import TSNE

data_directory = '../data/A1400/A1407/'

info = pd.read_csv(data_directory+'A1407.csv')
info = info.set_index('Session')

sessions = os.listdir(data_directory)
sessions.remove('A1407.csv') 
sessions = np.sort(sessions)

density = pd.DataFrame(index = sessions, columns = np.arange(4), data = 0.0)
hd_total = pd.DataFrame(index = sessions, columns = np.arange(4), data = 0.0)
hd_neurons = []

for s in sessions:
	path 								= os.path.join(data_directory, s)
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	episodes 							= info.filter(like='Trial').loc[s].dropna().values
	events								= list(np.where(episodes == 'wake')[0].astype('str'))
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)

	tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)

	tokeep, stat 						= findHDCells(tuning_curves[1])

	index 								= np.array([s+'_'+str(k) for k in spikes])

	hd_neurons.append(index[tokeep])

	for k in np.unique(shank):
		density.loc[s, k] 				= np.sum(shank == k)
		hd_total.loc[s,k] 					= np.sum(shank[tokeep] == k)




space = 0.05
x = np.arange(0.0, 4*0.2, 0.2)
y = info.loc[sessions,'Depth'].values*1e-3
y = np.cumsum(y)-y[0]
y[4] = y[3]+0.01

xnew, ynew, xytotal = interpolate(density.values.copy(), x, y, space)
xnew, ynew, hdtotal = interpolate(hd_total.values.copy(), x, y, space)


figure()
subplot(121)
imshow(xytotal)
subplot(122)
imshow(hdtotal)
show()
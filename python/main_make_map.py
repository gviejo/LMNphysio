import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys, os
from sklearn.manifold import TSNE

data_directory = '../data/A1400/A1407/'

# data_directory	= '/mnt/DataGuillaume/LMN/A1407/'

info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)


# sessions = os.listdir(data_directory)
# sessions.remove('A1407.csv') 
# sessions = np.sort(sessions)

sessions = info.loc['A1407-190403':].index.values

sessions = np.delete(sessions, np.where(sessions=='A1407-190406')[0])


density = pd.DataFrame(index = sessions, columns = np.arange(4), data = 0.0)
hd_total = pd.DataFrame(index = sessions, columns = np.arange(4), data = 0.0)
hd_neurons = []

alltcurves = []

for s in sessions:
	path 								= os.path.join(data_directory, s)
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	episodes 							= info.filter(like='Trial').loc[s].dropna().values
	events								= list(np.where(episodes == 'wake')[0].astype('str'))
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)

	# tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
	tcurves 							= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
	# tuning_curves[1] 					= smoothAngularTuningCurves(tuning_curves[1], window = 20, deviation = 3.0)
	tcurves2 						 	= smoothAngularTuningCurves(tcurves.copy(), window = 20, deviation = 3.0)	
	# tokeep, stat 						= findHDCells(tuning_curves[1])
	tokeep, stat 						= findHDCells(tcurves)

	# figure()
	# # for i,n in enumerate(tcurves.columns):
	# for i, n in enumerate(tokeep):
	# 	subplot(5,6,i+1,projection='polar')
	# 	plot(tcurves[n])
	# 	plot(tcurves2[n])


	index 								= np.array([s+'_'+str(k) for k in spikes])

	hd_neurons.append(index[tokeep])

	tcurves2.columns = pd.Index(index)

	if len(index[tokeep]):
		alltcurves.append(tcurves2[index[tokeep]])

	for k in np.unique(shank):
		density.loc[s, k] 				= np.sum(shank == k)
		hd_total.loc[s,k] 				= np.sum(shank[tokeep] == k)


alltcurves = pd.concat(alltcurves, 1)
alltcurves.to_hdf('../figures/figures_poster_2019/alltcurves.h5', 'w')

space = 0.01
x = np.arange(0.0, 4*0.2, 0.2)
y = info.loc[sessions,'Depth'].values*1e-3
y = np.cumsum(y)-y[0]
idx = np.where(np.diff(y) == 0)[0]
for i in idx+1: y[i] = y[i-1] + 0.001

xnew, ynew, xytotal = interpolate(density.values.copy(), x, y, space)
xnew, ynew, hdtotal = interpolate(hd_total.values.copy(), x, y, space)

xpos, ypos = np.meshgrid(x, y)

def noaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.set_xticks([])
	ax.set_yticks([])
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)


figure()
subplot(121)
imshow(xytotal, interpolation='gaussian')
subplot(122)
imshow(hdtotal, interpolation='gaussian')
show()

figure()
# noaxis(gca())
gca().invert_yaxis()
[plot([xpos[0,i], xpos[-1,i]], [ypos[0,i]-5, ypos[-1,i]], alpha = 0.7, color = 'lightgrey', zorder = 0, linewidth = 4) for i in range(4)]
scatter(xpos, ypos, s = 10, color = 'darkgray', alpha = 0.7)
scatter(xpos, ypos, s = hd_total*15, color = 'red', zorder = 3, alpha = 0.7)
xlim(-1, 1)
gca().set_aspect('equal')

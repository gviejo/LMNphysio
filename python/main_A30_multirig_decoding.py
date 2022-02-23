import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pingouin import partial_corr
from umap import UMAP
import sys
from matplotlib.colors import hsv_to_rgb
import hsluv
from sklearn.manifold import Isomap

def zscore_rate(rate):
	idx = rate.index
	cols = rate.columns
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	rate = pd.DataFrame(index = idx, data = rate, columns = cols)
	return nts.TsdFrame(rate)


data_directory = '/mnt/Data2/PSB/A8608/A8608-220106'


episodes = ['sleep', 'wake', 'wake', 'sleep', 'wake', 'wake', 'sleep']
events = ['1', '2', '4', '5']




spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)


position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
# sws_ep								= loadEpoch(data_directory, 'sws')
# rem_ep 								= loadEpoch(data_directory, 'rem')

#################
# TUNING CURVES
tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 120)
#tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tuning_curves 						= smoothAngularTuningCurves(tuning_curves, 10, 2)

tokeep, stat 						= findHDCells(tuning_curves, z=1, p = 0.001)

# mean_fr 							= computeMeanFiringRate(spikes, [wake_ep], ['wake'])
# tokeep 								= mean_fr.index.values[np.where(mean_fr>1)[0]]


figure()
count = 1
for i, n in enumerate(np.where(shank == 1)[0]):
	subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
	plot(tuning_curves[n])
	xticks([])
	yticks([])
	title(n)
	count += 1


###############
# Binning
wak_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, wake_ep, 300, 3))

corr_neurons = []

for i in range(len(wake_ep)):
	new_ep = refineWakeFromAngularSpeed(position['ry'], wake_ep.loc[[i]], bin_size = 300)
	tmp = np.corrcoef(wak_rate.restrict(new_ep).values.T)
	corr_neurons.append(tmp[np.triu_indices_from(tmp, 1)])

corr_neurons = np.array(corr_neurons).T

corr_sess = np.corrcoef(corr_neurons.T)

###############
# Binning




#sys.exit()
# tcurves 							= tuning_curves[tokeep]
# peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
# tcurves 							= tcurves[peaks.index.values]
tc_all = []
for i in range(len(wake_ep)):
	tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[i]], 60)
	tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)
	tuning_curves = tuning_curves/tuning_curves.max()
	tc_all.append(tuning_curves)

figure()
count = 1
for i, n in enumerate(np.where(shank == 5)[0]):
	subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
	for j in range(len(wake_ep)):
	#for j in [2,3]:
		plot(tc_all[j][n])
	xticks([])
	yticks([])
	title(n)
	count += 1

sys.exit()

#neurons = [3,4,5,7,10,13,20,23,24,27,29,35,39,50,59,63]
# neurons = [3,4,5,7,10,12,16,18,20,21,23,24,26,27,29,31,32,35,37,38,39,41,50,52,59,62,63]
neurons = np.where(shank==1)[0]

bin_size = 300

data = []
sessions = []
angles = []
for e in [0,1,2,3]:
	ep = wake_ep.loc[[e]]
	bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)

	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
	for i in neurons:
		spks = spikes[i].as_units('ms').index.values
		spike_counts[i], _ = np.histogram(spks, bins)

	rate = np.sqrt(spike_counts/(bin_size*1e-3))
	#rate = spike_counts/(bin_size*1e-3)

	rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=5)

	rate = nts.TsdFrame(t = rate.index.values, d = rate.values, time_units = 'ms')

	new_ep = refineWakeFromAngularSpeed(position['ry'], ep, bin_size = 300, thr = 0.1)

	rate = rate.restrict(new_ep)

	# angle = position['ry'].restrict(ep)
	# wakangle = pd.Series(index = np.arange(len(bins)-1), dtype = np.float)
	# tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
	# wakangle.loc[tmp.index] = tmp
	# wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)
	# wakangle = nts.Tsd(t = wakangle.index.values, d = wakangle.values, time_units = 'ms')
	# wakangle = wakangle.restrict(new_ep)

	# H = wakangle.values/(2*np.pi)
	# HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
	# RGB = hsv_to_rgb(HSV)

	# cutting the 20th percentile
	tmp = rate.values
	index = tmp.sum(1) > np.percentile(tmp.sum(1), 20)

	tmp = tmp[index,:]
#	RGB = RGB[index,:]

	
	# ump = UMAP(n_neighbors = 500, min_dist = 1).fit_transform(tmp)
	# scatter(ump[:,0], ump[:,1], c = RGB)
	# show()
	# break


	data.append(tmp)
	sessions.append(np.ones(len(tmp))*e)
	#angles.append(RGB)


data = np.vstack(data)
sessions = np.hstack(sessions)
#angles = np.vstack(angles)

umps = {}

for i, j in combinations(range(4), 2):
	idx = np.logical_or(sessions == i, sessions == j)
	ump = UMAP(n_neighbors = 600, min_dist = 1).fit_transform(data[idx,:])
	umps[(i,j)] = ump

for i in range(4):
	idx = sessions == i
	ump = UMAP(n_neighbors = 600, min_dist = 1).fit_transform(data[idx,:])
	umps[(i,i)] = ump


mkrs = ['o', 's']
colors = ['blue', 'green', 'red', 'yellow']
order = ['circle', 'square', 'circle', 'square']
size = 5
figure()
gs = GridSpec(4,4)
for i, j in combinations(range(4), 2):
	subplot(gs[i,j])
	idx = sessions[np.logical_or(sessions == i, sessions == j)]
	ump = umps[(i,j)]
	for k, n in enumerate(np.unique(idx)):
		scatter(ump[idx==n,0], ump[idx==n,1], s = size, c = "None", edgecolors = colors[int(n)], alpha = 0.5)
	title("/".join([order[i],order[j]]))

for i in range(4):
	subplot(gs[i,i])
	scatter(umps[(i,i)][:,0], umps[(i,i)][:,1], s = size, c = "None", edgecolors = colors[i], alpha = 0.5)

show()

# from mpl_toolkits.mplot3d import Axes3D

#ump = Isomap(n_components = 3, n_neighbors = 100).fit_transform(data)
# fig = figure()
# mkrs = ['o', 's']
# colors = ['blue', 'green']
# ax = fig.add_subplot(111, projection='3d')
# for i, n in enumerate(np.unique(sessions)):
# 	ax.scatter(ump[sessions==n,0], ump[sessions==n,1], ump[sessions==n,2], color = colors[i])#c= angles[sessions==n], marker = mkrs[i])#, alpha = 0.5, linewidth = 0, s = 100)

# show()
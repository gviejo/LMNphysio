import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec
from umap import UMAP
import sys
from matplotlib.colors import hsv_to_rgb
import hsluv
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D


def zscore_rate(rate):
	idx = rate.index
	cols = rate.columns
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	rate = pd.DataFrame(index = idx, data = rate, columns = cols)
	return nts.TsdFrame(rate)


data_directory = '/mnt/Data2/PSB/A8608/A8608-220108'


episodes = ['sleep', 'wake', 'wake', 'wake', 'sleep', 'wake', 'wake', 'wake', 'sleep']
events = ['1', '2', '3', '5', '6', '7']




spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)


position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					

tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 120)
tuning_curves 						= smoothAngularTuningCurves(tuning_curves, 10, 2)
tokeep, stat 						= findHDCells(tuning_curves, z=1, p = 0.001)

spatial_info = computeSpatialInfo(tuning_curves, position['ry'], wake_ep.loc[[0]])

neurons = spatial_info['SI'][spatial_info['SI']>0.2].index.values

idx = spatial_info['SI'].sort_values().dropna().index.values

figure()
count = 1
# for i, n in enumerate(np.where(shank == 1)[0]):
for i, n in enumerate(idx):
	subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
	plot(tuning_curves[n])
	if n in neurons:
	 	plot(tuning_curves[n], linewidth = 3)
	xticks([])
	yticks([])
	title(str(n) + ' ' + str(np.round(spatial_info.loc[n,'SI'], 3)))
	count += 1

show()
# sys.exit()


bin_size = 300

data = []
sessions = []
angles = []
for e in [3,5]:
	ep = wake_ep.loc[[e]]
	bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)

	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
	for i in neurons:
		spks = spikes[i].as_units('ms').index.values
		spike_counts[i], _ = np.histogram(spks, bins)

	rate = np.sqrt(spike_counts/(bin_size*1e-3))
	#rate = spike_counts/(bin_size*1e-3)

	rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)

	rate = nts.TsdFrame(t = rate.index.values, d = rate.values, time_units = 'ms')

	new_ep = refineWakeFromAngularSpeed(position['ry'], ep, bin_size = 300, thr = 0.1)
	rate = rate.restrict(new_ep)

	angle = position['ry'].restrict(ep)
	wakangle = pd.Series(index = np.arange(len(bins)-1), dtype = np.float)
	tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
	wakangle.loc[tmp.index] = tmp
	wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)
	wakangle = nts.Tsd(t = wakangle.index.values, d = wakangle.values, time_units = 'ms')
	wakangle = wakangle.restrict(new_ep)

	H = wakangle.values/(2*np.pi)
	HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
	RGB = hsv_to_rgb(HSV)

	# cutting the 20th percentile
	tmp = rate.values
	index = tmp.sum(1) > np.percentile(tmp.sum(1), 20)
	tmp = tmp[index,:]
	RGB = RGB[index,:]

	# ump = Isomap(n_components = 3, n_neighbors = 100).fit_transform(tmp)	
	# # ump = UMAP(n_neighbors = 500, min_dist = 1).fit_transform(tmp)
	# scatter(ump[:,0], ump[:,1], c = RGB)
	# show()
	# sys.exit()

	data.append(tmp)
	sessions.append(np.ones(len(tmp))*e)
	angles.append(RGB)


data = np.vstack(data)
sessions = np.hstack(sessions)
angles = np.vstack(angles)


ump = Isomap(n_components = 3, n_neighbors = 100).fit_transform(data)

fig = figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(ump[:,0], ump[:,1], ump[:,2], c = sessions)
show()

sys.exit()

umps = {}
rgbs = {}

for i, j in combinations(range(4), 2):
	idx = np.logical_or(sessions == i, sessions == j)
	# ump = UMAP(n_neighbors = 600, min_dist = 1).fit_transform(data[idx,:])
	ump = Isomap(n_components = 3, n_neighbors = 100).fit_transform(data[idx,:])
	umps[(i,j)] = ump
	rgbs[(i,j)] = angles[idx,:]

for i in range(4):
	idx = sessions == i
	ump = Isomap(n_components = 3, n_neighbors = 100).fit_transform(data[idx,:])
	umps[(i,i)] = ump
	rgbs[(i,i)] = angles[idx,:]

sys.exit()

mkrs = ['o', 's']
colors = ['blue', 'green', 'red', 'yellow']
# order = ['circle', 'square', 'circle', 'square']
order = np.array(['square', '8 arm maze', 'square', '8 arm maze'])
size = 5

# (0, 1) and (2,3)



fig = figure()
for i,p in enumerate([(0,1), (2,3)]):
	ax = fig.add_subplot(2,2,i+1, projection='3d')
	ax.scatter(umps[p][:,0], umps[p][:,1], umps[p][:,2], color = rgbs[p])
	ax.set_title("-".join(order[list(p)]))
	print(i+1,i+2)
	ax = fig.add_subplot(2,2,i+3, projection='3d')
	idx = np.logical_or(sessions == p[0], sessions == p[1])
	ax.scatter(umps[p][:,0], umps[p][:,1], umps[p][:,2], c = sessions[idx])
show()

# (0, 2) and (1,3)

fig = figure()
for i,p in enumerate([(0,2), (1,3)]):
	ax = fig.add_subplot(2,2,i+1, projection='3d')
	ax.scatter(umps[p][:,0], umps[p][:,1], umps[p][:,2], color = rgbs[p])
	ax.set_title("-".join(order[list(p)]))
	print(i+1,i+2)
	ax2 = fig.add_subplot(2,2,i+3, projection='3d')
	idx = np.logical_or(sessions == p[0], sessions == p[1])
	ax2.scatter(umps[p][:,0], umps[p][:,1], umps[p][:,2], c = sessions[idx])
show()

# (0, 3) and (1,2)

fig = figure()
for i,p in enumerate([(0,3), (1,2)]):
	ax = fig.add_subplot(2,2,i+1, projection='3d')
	ax.scatter(umps[p][:,0], umps[p][:,1], umps[p][:,2], color = rgbs[p])
	ax.set_title("-".join(order[list(p)]))
	print(i+1,i+2)
	ax2 = fig.add_subplot(2,2,i+3, projection='3d')
	idx = np.logical_or(sessions == p[0], sessions == p[1])
	ax2.scatter(umps[p][:,0], umps[p][:,1], umps[p][:,2], c = sessions[idx])
show()


figure()
for i in range(4):
	subplot(1,4,i+1)
	plot(position['x'].restrict(wake_ep.loc[[i]]), position['z'].restrict(wake_ep.loc[[i]]))


ump = Isomap(n_components = 3, n_neighbors = 100).fit_transform(data)

fig = figure()
ax = subplot(111, projection='3d')
ax.scatter(ump[:,0], ump[:,1], ump[:,2], c=sessions)

for i,p in enumerate([(0,3), (1,2)]):
	ax = fig.add_subplot(2,2,i+1, projection='3d')
	ax.scatter(umps[p][:,0], umps[p][:,1], umps[p][:,2], color = rgbs[p])
	ax.set_title("-".join(order[list(p)]))
	print(i+1,i+2)
	ax2 = fig.add_subplot(2,2,i+3, projection='3d')
	idx = np.logical_or(sessions == p[0], sessions == p[1])
	ax2.scatter(umps[p][:,0], umps[p][:,1], umps[p][:,2], c = sessions[idx])
show()

# figure()
# gs = GridSpec(4,4)
# for i, j in combinations(range(4), 2):
# 	subplot(gs[i,j])
# 	idx = sessions[np.logical_or(sessions == i, sessions == j)]
# 	ump = umps[(i,j)]
# 	for k, n in enumerate(np.unique(idx)):
# 		scatter(ump[idx==n,0], ump[idx==n,1], s = size, c = "None", edgecolors = colors[int(n)], alpha = 0.5)
# 	title("/".join([order[i],order[j]]))

# for i in range(4):
# 	subplot(gs[i,i])
# 	scatter(umps[(i,i)][:,0], umps[(i,i)][:,1], s = size, c = "None", edgecolors = colors[i], alpha = 0.5)

# show()



# #ump = Isomap(n_components = 3, n_neighbors = 100).fit_transform(data)
# # fig = figure()
# # mkrs = ['o', 's']
# # colors = ['blue', 'green']
# # ax = fig.add_subplot(111, projection='3d')
# # for i, n in enumerate(np.unique(sessions)):
# # 	ax.scatter(ump[sessions==n,0], ump[sessions==n,1], ump[sessions==n,2], color = colors[i])#c= angles[sessions==n], marker = mkrs[i])#, alpha = 0.5, linewidth = 0, s = 100)

# # show()
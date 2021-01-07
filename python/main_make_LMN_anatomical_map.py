import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from matplotlib.colors import hsv_to_rgb
import hsluv
from pycircstat.descriptive import mean as circmean
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.cluster import KMeans
from umap import UMAP
from matplotlib import gridspec

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

datasets = ['LMN-ADN/A5002/'+s for s in infos['A5002'].index[1:-5]]

ahv = []
mapping = []
autocorr = {e:[] for e in ['wak', 'rem', 'sws']}
isis = {e:[] for e in ['wak', 'rem', 'sws']}
frates = {e:[] for e in ['wak', 'rem', 'sws']}

for s in datasets:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)	
	sws_ep			= loadEpoch(path, 'sws')
	rem_ep			= loadEpoch(path, 'rem')
	meanwavef, maxch = loadMeanWaveforms(path)

	# TO RESTRICT BY SHANK
	spikes 			= {n:spikes[n] for n in np.where(shank>2)[0]}
	neurons 		= [name+'_'+str(n) for n in spikes.keys()]

	######################
	# TUNING CURVEs
	######################
	tcurves 		= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)	
	tcurves 		= smoothAngularTuningCurves(tcurves, 20, 2)
	tokeep, stat 	= findHDCells(tcurves)

	######################
	# AUTOCORR
	######################
	for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):
		corr, frt = compute_AutoCorrs(spikes, ep, 2, 400)
		corr.columns = pd.Index(neurons)
		frt.index = pd.Index(neurons)		
		autocorr[e].append(corr)
		frates[e].append(frt)
	
	######################
	# ISI
	######################
	for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):
		isi = compute_ISI(spikes, ep, 10000, 400, True)
		isi.columns = pd.Index(neurons)
		isis[e].append(isi)

	######################
	# AHV
	######################
	velo_curves 	= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30, norm=True)
	velo_curves = velo_curves.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1)
	velo_curves.columns = pd.Index(neurons)

	######################
	# POSITION
	######################
	mapp = pd.DataFrame(index = neurons,
		columns = ['session', 'shank', 'channel', 'hd'], 
		data = datasets.index(s))
	mapp['shank'] = shank[shank>2]
	mapp['channel'] = maxch[list(spikes.keys())].values
	mapp['hd'] = 0
	mapp.loc[[name+'_'+str(n) for n in tokeep],'hd'] = 1

	######################
	ahv.append(velo_curves)
	mapping.append(mapp)
	


mapping = pd.concat(mapping)
ahv = pd.concat(ahv, 1)
for e in autocorr.keys():
	autocorr[e] = pd.concat(autocorr[e], 1)
	isis[e] = pd.concat(isis[e], 1)
	frates[e] = pd.concat(frates[e])

frates = pd.DataFrame.from_dict(frates)

# NEURONS POSITION
mapping['x'] = 0
mapping['y'] = 0
x = infos['A5002']['LMN'].loc[[s.split('/')[-1] for s in datasets]].values
x[x==0] = np.ones(np.sum(x==0))
x = np.cumsum(x) + 20*16
x = pd.Series(index = [s.split('/')[-1] for s in datasets], data = x)
for n in mapping.index:
	mapping.loc[n,'x'] = (x[n.split('_')[0]] - 20*(16-mapping.loc[n,'channel']))*-1
	mapping.loc[n,'y'] = mapping.loc[n,'shank']*250 + np.random.uniform(-40, 40)


# CLUSTER AHV
# ahv = ahv - ahv.mean(0)
# ahv = ahv - ahv.mean(0)
# ahv = ahv / ahv.std(0)
# ump = UMAP(n_neighbors = 10, min_dist = 0.001).fit_transform(ahv.values.T)
# km = KMeans(2).fit(ump).labels_

# # scatter(ump[:,0], ump[:,1], c = km)


# CLUSTER AUTOCORR
data = []
for e in autocorr.keys():
	tmp = autocorr[e].loc[0.5:]
	# tmp = tmp.drop(tmp.columns[tmp.apply(lambda col: col.max() > 200.0)], axis = 1)
	tmp = tmp.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
	# tmp = tmp.dropna(1)	
	data.append(tmp.loc[0:100])

# neurons = np.intersect1d(data[0].columns, np.intersect1d(data[1].columns, data[2].columns))
neurons = frates[(frates>=1).prod(1) == 1].index

data = np.vstack([d[neurons] for d in data])

ump = UMAP(n_neighbors = 40, min_dist = 0.1).fit_transform(data.T)
km = KMeans(5).fit(ump).labels_

figure()
gs = gridspec.GridSpec(4,1+len(np.unique(km)))
subplot(gs[0,0])
cmap = matplotlib.cm.get_cmap('jet')
clrs = cmap(km/np.max(km))
ax = scatter(ump[:,0], ump[:,1], c = clrs)
subplot(gs[1,0])
scatter(mapping['y'], mapping['x'], c = mapping['hd'])


for i, k in enumerate(np.unique(km)):
	subplot(gs[0,i+1])
	ax2 = scatter(mapping.loc[neurons,'y'], mapping.loc[neurons, 'x'], c = 'grey', alpha = 0.5)	
	ax3 = scatter(mapping.loc[neurons[km==k],'y'], mapping.loc[neurons[km==k],'x'], c = clrs[km==k])

	for j, e in enumerate(autocorr.keys()):
		subplot(gs[j+1,i+1])
		plot(autocorr[e][neurons[km==k]].loc[-200:200], color = 'grey', alpha = 0.3)
		plot(autocorr[e][neurons[km==k]].mean(1).loc[-200:200], color = cmap(k/np.max(km)), linewidth = 4)



sys.exit()
# CLUSTER ISI
data = []
for e in isis.keys():
	tmp = isis[e]	
	tmp = tmp.rolling(window = 40, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 2.0)
	tmp = tmp.dropna(1)	
	data.append(tmp)
neurons = np.intersect1d(data[0].columns, np.intersect1d(data[1].columns, data[2].columns))
data = np.vstack([d[neurons] for d in data])

ump = UMAP(n_neighbors = 5, min_dist = 0.0001).fit_transform(data.T)
km = KMeans(5).fit(ump).labels_

figure()
gs = gridspec.GridSpec(3,1+len(np.unique(km)))
subplot(gs[0,0])
cmap = matplotlib.cm.get_cmap('jet')
clrs = cmap(km/np.max(km))
ax = scatter(ump[:,0], ump[:,1], c = clrs)
subplot(gs[1,0])
scatter(mapping['y'], mapping['x'], c = mapping['hd'])
for i, k in enumerate(np.unique(km)):
	subplot(gs[0,i+1])
	ax2 = scatter(mapping.loc[neurons,'y'], mapping.loc[neurons, 'x'], c = 'grey', alpha = 0.5)	
	ax3 = scatter(mapping.loc[neurons[km==k],'y'], mapping.loc[neurons[km==k],'x'], c = clrs[km==k])

	subplot(gs[1,i+1])
	plot(ahv[neurons[km==k]], color = cmap(k/np.max(km)), alpha = 0.5)
	plot(ahv[neurons[km==k]].mean(1), color = cmap(k/np.max(km)))





sys.exit()
# DENSITY CLUSTER AUTOCORR
kcount = np.zeros((len(np.unique(km)),len(datasets),4))
for k in range(len(kcount)):
	for i in range(len(datasets)):
		for j in range(4):
			mps = mapping.iloc[km==k]
			idx = mps[(mps['session']==i)&(mps['shank']==j+3)].index
			kcount[k,i,j] = len(idx)

figure()
gs = gridspec.GridSpec(2,len(kcount))
for k in range(len(kcount)):
	subplot(gs[0,k])
	imshow(kcount[k])
	subplot(gs[1,k])
	plot(autocorr.iloc[:,km==k], color = 'grey', alpha = 0.3)
	plot(autocorr.iloc[:,km==k].mean(1), linewidth = 3)



# NEURON DENSITY
count = np.zeros((len(datasets), 4))
for i in range(len(datasets)):
	for j in range(4):
		count[i,j] = len(mapping[(mapping['session']==i)&(mapping['shank']==j+3)].index)
x = infos['A5002']['LMN'].loc[[s.split('/')[-1] for s in datasets]].values
x[x==0] = np.ones(np.sum(x==0))
x = np.cumsum(x)
y = np.arange(0, 250*4, 250)
yy, xx = np.meshgrid(y, x)
f = scipy.interpolate.interp2d(yy, xx, count, kind='cubic')
bs = 25
xnew = np.arange(x[0]-2*bs, x[-1]+3*bs, bs)
ynew = np.arange(y[0]-2*bs, y[-1]+3*bs, bs)
znew = f(ynew, xnew)



figure()
subplot(121)
imshow(znew.T)
subplot(122)
imshow(count)



from matplotlib import gridspec

for k in np.unique(km):
	figure()
	gs = gridspec.GridSpec(len(datasets), 4)
	for i in range(len(datasets)):
		for j in range(4):
			subplot(gs[i,j])
			mps = mapping.iloc[km==k]
			idx = mps[(mps['session']==i)&(mps['shank']==j+3)].index
			if len(idx):
				plot(ahv[idx])



for k in np.unique(km):
	figure()
	gs = gridspec.GridSpec(len(datasets), 4)
	for i in range(len(datasets)):
		for j in range(4):
			subplot(gs[i,j])
			mps = mapping.iloc[km==k]
			idx = mps[(mps['session']==i)&(mps['shank']==j+3)].index
			if len(idx):
				plot(autocorr[idx])
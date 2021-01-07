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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
from matplotlib import gridspec

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

# A5002
datasets = ['LMN-ADN/A5002/'+s for s in infos['A5002'].index[1:-1]]
datasets.remove('LMN-ADN/A5002/A5002-200306A')

# A1407
# datasets = ['LMN/A1407/'+s for s in infos['A1407'].index[10:-2]]
# datasets.remove('LMN/A1407/A1407-190406')


mapping = []
ccall = []
autocorr = {e:[] for e in ['wak', 'rem', 'sws']}
frates = {e:[] for e in ['wak', 'rem', 'sws']}
alltcurves = []

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
	sys.exit()
	# TO RESTRICT BY SHANK
	if 'A5002' in s:
		spikes 			= {n:spikes[n] for n in np.where(shank>2)[0]}

	
	neurons 		= [name+'_'+str(n) for n in spikes.keys()]

	######################
	# TUNING CURVEs
	######################
	tcurves 		= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)	
	tcurves 		= smoothAngularTuningCurves(tcurves, 20, 2)
	tokeep, stat 	= findHDCells(tcurves)
	peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
	tcurves.columns						= pd.Index(neurons)
	alltcurves.append(tcurves)	
	
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
	# POSITION
	######################
	mapp = pd.DataFrame(index = neurons,
		columns = ['session', 'shank', 'channel', 'hd', 'peak'], 
		data = datasets.index(s))
	if 'A5002' in s:
		mapp['shank'] = shank[shank>2]
	else:
		mapp['shank'] = shank

	mapp['channel'] = maxch[list(spikes.keys())].values
	mapp['hd'] = 0
	mapp.loc[[name+'_'+str(n) for n in tokeep],'hd'] = 1	
	mapp.loc[[name+'_'+str(n) for n in peaks.index],'peak'] = peaks.values
	mapping.append(mapp)
	
	######################
	# SYNAPTIC CONNECTION
	######################

	spks = spikes
	binsize = 0.5
	nbins = 100
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

	from itertools import product

	cc = pd.DataFrame(index = times, columns = list(product(neurons, neurons)))
	# cc2 = pd.DataFrame(index = times, columns = list(product(spikes.keys(), spikes.keys())))
		
	for i,j in cc.columns:
		if i != j:
			spk1 = spks[int(i.split("_")[1])].as_units('ms').index.values
			spk2 = spks[int(j.split("_")[1])].as_units('ms').index.values		
			tmp = crossCorr(spk1, spk2, binsize, nbins)					
			cc[(i,j)] = tmp			
	cc = cc.dropna(1)
	
	# CONVOLUTION WITH HOLLOW GAUSSIAN KERNEL
	ccslow = cc.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 10)
	cc2 = cc - ccslow

	ccall.append(cc)

mapping = pd.concat(mapping)

ccall = pd.concat(ccall,1)
for e in autocorr.keys():
	autocorr[e] = pd.concat(autocorr[e], 1)	
	frates[e] = pd.concat(frates[e])
frates = pd.DataFrame.from_dict(frates)
alltcurves 	= pd.concat(alltcurves, 1)

###########################################################################################
# NEURONS POSITION
###########################################################################################
mapping['x'] = 0
mapping['y'] = 0

n_shank = len(shank_to_channel[2])

x = infos[datasets[0].split('/')[1]]['LMN'].loc[[s.split('/')[-1] for s in datasets]].values
x[x==0] = np.ones(np.sum(x==0))
x = np.cumsum(x) + 20*n_shank
x = pd.Series(index = [s.split('/')[-1] for s in datasets], data = x)
for n in mapping.index:
	mapping.loc[n,'x'] = (x[n.split('_')[0]] - 20*(n_shank-mapping.loc[n,'channel']))*-1
	mapping.loc[n,'y'] = mapping.loc[n,'shank']*250 + np.random.uniform(-40, 40)


# SYNAPTIC DETECTION
ccfast = ccall.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 2)
ccslow = ccall.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 10)

cc = (ccfast - ccslow)/ccfast.std(0)
idx = cc.loc[0:3].mean(0).sort_values().index.values
cc = cc[idx]

pairs = idx[-int(len(idx)*0.01):]
pairs2 = idx[:int(len(idx)*0.01)]


figure()
subplot(231)
for i,j in pairs:
	plot(mapping.loc[[i,j],['y']], mapping.loc[[i,j],['x']], linewidth = 1, alpha = 0.5, color = 'grey')
plot(mapping['y'], mapping['x'], 'o', markersize = 2)	

subplot(232)
plot(cc[pairs])

subplot(234)
for i,j in pairs2:
	plot(mapping.loc[[i,j],['y']], mapping.loc[[i,j],['x']], linewidth = 1, alpha = 0.5, color = 'grey')
plot(mapping['y'], mapping['x'], 'o', markersize = 2)	

subplot(235)
plot(cc[pairs2])

subplot(1,3,3)
imshow(cc.values.T, aspect = 'auto')
axhline(len(pairs2))
axhline(cc.shape[1] - len(pairs))


###########################################################################################
# CLUSTER AUTOCORR
###########################################################################################
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
km = KMeans(6).fit(ump).labels_

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


###########################################################################################
# HD NEURONS
###########################################################################################
hd_neurons = mapping[mapping['hd']==1].index.values
hd_pairs = [p for p in pairs if p[0] in hd_neurons and p[1] in hd_neurons]
hd_pairs2 = [p for p in pairs2 if p[0] in hd_neurons and p[1] in hd_neurons]
hdp_neurons = np.unique(np.array(hd_pairs).flatten())

H = mapping.loc[hd_neurons, 'peak'].values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)
RGB = pd.DataFrame(index = hd_neurons, data = RGB)

figure()
subplot(221)
scatter(mapping['y'], mapping['x'], color = 'grey', alpha = 0.5, ec = None)
scatter(mapping.loc[hd_neurons,'y'], mapping.loc[hd_neurons,'x'], c = RGB.values)
scatter(mapping.loc[hdp_neurons,'y'], mapping.loc[hdp_neurons,'x'], c = 'white', s = 1)
subplot(222)
plot(np.cos(mapping.loc[hd_neurons,'peak']), np.sin(mapping.loc[hd_neurons,'peak']), '.', color = 'grey')

prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
            shrinkA=0,shrinkB=0, color = 'grey', alpha = 0.4)
for p in hd_pairs:
	# plot(np.cos(mapping.loc[list(p),'peak']), np.sin(mapping.loc[list(p),'peak']), '-', alpha = 0.5, color = 'grey')
	xystart = np.array([np.cos(mapping.loc[p[1],'peak']),np.sin(mapping.loc[p[1],'peak'])])
	xyend = np.array([np.cos(mapping.loc[p[0],'peak']),np.sin(mapping.loc[p[0],'peak'])])
	dxy = xyend - xystart
	xyend = xystart + 0.9*dxy
	annotate('', xyend, xystart, arrowprops=prop)

subplot(224)
hd_pairs_wtrep = [p for p in hd_pairs if (p[1],p[0]) not in list(hd_pairs)]

plot(np.cos(mapping.loc[hd_neurons,'peak']), np.sin(mapping.loc[hd_neurons,'peak']), '.', color = 'grey')
prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
            shrinkA=0,shrinkB=0, color = 'grey', alpha = 0.4)
for p in hd_pairs_wtrep:
	# plot(np.cos(mapping.loc[list(p),'peak']), np.sin(mapping.loc[list(p),'peak']), '-', alpha = 0.5, color = 'grey')
	xystart = np.array([np.cos(mapping.loc[p[1],'peak']),np.sin(mapping.loc[p[1],'peak'])])
	xyend = np.array([np.cos(mapping.loc[p[0],'peak']),np.sin(mapping.loc[p[0],'peak'])])
	dxy = xyend - xystart
	xyend = xystart + 0.9*dxy
	annotate('', xyend, xystart, arrowprops=prop)

subplot(223, projection = 'polar')
alpha = np.array([mapping.loc[list(p),'peak'].diff().values[-1] for p in hd_pairs])
alpha[alpha>np.pi] = 2*np.pi - alpha[alpha>np.pi]
alpha[alpha<-np.pi] = 2*np.pi + alpha[alpha<-np.pi]
bins = np.linspace(-np.pi, np.pi, 24)
x,_ = np.histogram(alpha, bins)
tmp = mapping.loc[hd_neurons, 'peak'].values
tmp = np.vstack(tmp) - tmp
tmp = tmp[np.triu_indices_from(tmp)]
tmp[tmp>np.pi] -= 2*np.pi
tmp[tmp<-np.pi] += 2*np.pi
yc,_ = np.histogram(tmp, bins)
# hist(alpha, bins)
plot(bins[0:-1], x/(yc+1))

# tmp = alltcurves[hd_neurons]
# tmp = tmp/tmp.max()

###########################################################################################
# NON HD NEURONS
###########################################################################################
hd_neurons 		= mapping[mapping['hd']==1].index.values
nonhd_pairs 	= [p for p in pairs if p[0] not in hd_neurons and p[1] not in hd_neurons]
nonhd_pairs2 	= [p for p in pairs2 if p[0] not in hd_neurons and p[1] not in hd_neurons]
nonhdp_neurons 	= np.unique(np.array(hd_pairs).flatten())

km2 = pd.Series(index = neurons, data = km)

connect_matrix = np.zeros((len(np.unique(km)), len(np.unique(km))))

for i, j in nonhd_pairs:
	if i in km2.index and j in km2.index:
		connect_matrix[km2[i],km2[j]] += 1 

connect_matrix[np.diag_indices_from(connect_matrix)] = 0

figure()
imshow(connect_matrix)


pairs_km = pd.DataFrame(index = nonhd_pairs, columns = ['i', 'j'])
for i, j in nonhd_pairs:
	if i in km2.index and j in km2.index:
		pairs_km.loc[(i,j), 'i'] = km2.loc[i]
		pairs_km.loc[(i,j), 'j'] = km2.loc[j]
pairs_km = pairs_km.dropna()

prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
		            shrinkA=0,shrinkB=0, color = 'grey', alpha = 0.6)

figure()
gs = gridspec.GridSpec(len(np.unique(km)), len(np.unique(km)))
for i in range(len(np.unique(km))):
	for j in range(len(np.unique(km))):
		subplot(gs[i,j])		
		scatter(mapping.loc[neurons[km==i],'y'], mapping.loc[neurons[km==i],'x'], c = clrs[km==i], ec = None, s = 5)
		scatter(mapping.loc[neurons[km==j],'y'], mapping.loc[neurons[km==j],'x'], c = clrs[km==j], ec = None, s = 5)
		xticks([])
		yticks([])
		tmp = pairs_km[(pairs_km['i'] == i)&(pairs_km['j'] == j)].index.values
		for p in tmp:			
			xystart = np.array([mapping.loc[p[0],'y'],mapping.loc[p[0],'x']])
			xyend = np.array([mapping.loc[p[1],'y'],mapping.loc[p[1],'x']])
			dxy = xyend - xystart
			xyend = xystart + 0.9*dxy
			annotate('', xyend, xystart, arrowprops=prop)
tight_layout()


figure()
subplot(221)
plot(cc[hd_pairs])
subplot(222)
plot(cc[hd_pairs2])
subplot(223)
plot(cc[nonhd_pairs])
subplot(224)
plot(cc[nonhd_pairs2])
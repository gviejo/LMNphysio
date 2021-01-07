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
from sklearn.cluster import KMeans, SpectralClustering
from umap import UMAP
from matplotlib import gridspec
from itertools import product


data_directory = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201014A'


episodes = ['sleep', 'wake', 'sleep']
events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
acceleration 						= acceleration[[0,1,2]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)
sws_ep 								= loadEpoch(data_directory, 'sws')
rem_ep 								= loadEpoch(data_directory, 'rem')


######################
# TUNING CURVEs
######################
tcurve 			= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)	
tcurve 			= smoothAngularTuningCurves(tcurve, 20, 2)
# CHECKING HALF EPOCHS
wake2_ep = splitWake(wake_ep)
tokeep2 = []
stats2 = []
tcurves2 = []
for i in range(2):
	# tcurves_half = computeLMNAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])[0][1]
	tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
	tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
	tokeep, stat = findHDCells(tcurves_half, 20)
	tokeep2.append(tokeep)
	stats2.append(stat)
	tcurves2.append(tcurves_half)

tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
tokeep2 = np.union1d(tokeep2[0], tokeep2[1])

adn = np.where(shank <=3)[0]
lmn = np.where(shank >3)[0]

adn = np.intersect1d(adn, tokeep)
lmn = np.intersect1d(lmn, tokeep)

spikes = {n:spikes[n] for n in tokeep}

peaks = pd.Series(index=tcurve.columns,data = np.array([circmean(tcurve.index.values, tcurve[i].values) for i in tcurve.columns])).sort_values()

for i, nn in enumerate([adn, lmn]):
	figure()
	for j, n in enumerate(nn):
		subplot(3,5,j+1, projection='polar')
		plot(tcurve[n])
		xticks([])
		title(n)



neurons = [7,23,29]

figure()
subplot(111,projection='polar')
plot(tcurve[neurons[0]], color = 'red')
[plot(tcurve[n], color = 'green') for n in neurons[1:]]

figure()
plot(position['ry'])
# adn
plot(spikes[neurons[0]].restrict(wake_ep).fillna(peaks[neurons[0]]), '|', markersize = 10, color = 'red')
[plot(spikes[n].restrict(wake_ep).fillna(peaks[n]), '|', markersize = 10, color = 'green') for n in neurons[1:]]

# MAKING EPOCHS OF CW, CCW and Forward epoch
tmp 			= pd.Series(index = position['ry'].index.values, data = np.unwrap(position['ry'].values))
tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
time_bins		= np.arange(tmp.index[0], tmp.index[-1]+10000, 10000) # assuming microseconds
index 			= np.digitize(tmp2.index.values, time_bins)
tmp3 			= tmp2.groupby(index).mean()
tmp3.index 		= time_bins[np.unique(index)-1]+np.diff(time_bins)/2
tmp3 			= nts.Tsd(tmp3)
tmp4			= np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
tmp2 			= nts.Tsd(tmp2)
tmp4			= np.diff(tmp2.values)/np.diff(tmp2.as_units('s').index.values)	
velocity 		= nts.Tsd(t=tmp2.index.values[1:], d = tmp4)
bins_velocity	= np.array([velocity.min(), -np.pi/6, np.pi/6, velocity.max()+0.001])


figure()
ax = subplot(211)
plot(position['ry'])
ax2 = subplot(212, sharex = ax)
plot(velocity)

velocity 		= velocity.as_series()

start = np.where(np.diff((velocity>(np.pi/10))*1) == 1)[0]
end = np.where(np.diff((velocity>(np.pi/10))*1) == -1)[0]
time_index = velocity.index.values[0:-1] + np.diff(velocity.index.values)/2
ccw_ep = nts.IntervalSet(start = time_index[start], end = time_index[end])

start = np.where(np.diff((velocity<(-np.pi/10))*1) == 1)[0]
end = np.where(np.diff((velocity<(-np.pi/10))*1) == -1)[0]
time_index = velocity.index.values[0:-1] + np.diff(velocity.index.values)/2
cw_ep = nts.IntervalSet(start = time_index[start], end = time_index[end])

figure()
velocity = nts.Tsd(velocity)
plot(velocity)
plot(velocity.restrict(ccw_ep), '.')
plot(velocity.restrict(cw_ep), '.')


# COMPUTING CC in each sub epochs
binsize = 1
nbins = 200
times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

cc_ccw = pd.DataFrame(index = times, columns = list(product(lmn, adn)))	
cc_cw = pd.DataFrame(index = times, columns = list(product(lmn, adn)))	

for i,j in cc_cw.columns:
	if i != j:
		spk1 = spikes[i].restrict(ccw_ep).as_units('ms').index.values
		spk2 = spikes[j].restrict(ccw_ep).as_units('ms').index.values		
		cc_ccw[(i,j)] = crossCorr(spk1, spk2, binsize, nbins)

		spk1 = spikes[i].restrict(cw_ep).as_units('ms').index.values
		spk2 = spikes[j].restrict(cw_ep).as_units('ms').index.values		
		cc_cw[(i,j)] = crossCorr(spk1, spk2, binsize, nbins)

ccahv = {}
for name, cc in zip(['ccw', 'cw'], [cc_ccw, cc_cw]):
	ccfast = cc.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 5)
	ccslow = cc.rolling(window=50, win_type='gaussian', center= True, min_periods=1).mean(std = 20)
	ccahv[name] = (cc - ccslow)/cc.std(0)
	

sys.exit()

for i in range(30,40):
	p = ccahv['ccw'].columns[i]
	figure()	
	subplot(121, projection = 'polar')
	plot(tcurve[p[0]], color = 'green')
	plot(tcurve[p[1]], color = 'red')
	subplot(122)
	for k in ccahv.keys():
		plot(ccahv[k].iloc[:,i])
	title(p)

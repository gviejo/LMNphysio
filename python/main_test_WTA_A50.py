import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from sklearn.ensemble import GradientBoostingClassifier
from pycircstat.descriptive import mean as circmean
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

def zscore_rate(rate):
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	return rate

############################################################################################### 
# GENERAL infos
###############################################################################################

data_directory = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201015A'

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
newsleep_ep 						= refineSleepFromAccel(acceleration, sleep_ep)
sws_ep 								= loadEpoch(data_directory, 'sws')

wake_ep 							= wake_ep.loc[[0]]

tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
tcurves 							= smoothAngularTuningCurves(tuning_curves, 10, 2)
tokeep, stat 						= findHDCells(tcurves)#, z=10, p = 0.001)



colors=dict(zip(np.unique(shank), cm.rainbow(np.linspace(0,1,len(np.unique(shank))))))
figure()
for i in tcurves.columns:
	subplot(int(np.ceil(np.sqrt(tcurves.shape[1]))),int(np.ceil(np.sqrt(tcurves.shape[1]))),i+1, projection='polar')
	plot(tcurves[i], color = colors[shank[i]])
	if i in tokeep:
		plot(tcurves[i], color = colors[shank[i]], linewidth = 4)



adn = np.intersect1d(tokeep, np.where(shank<=3)[0])
lmn = np.intersect1d(tokeep, np.where(shank==5)[0])

tcurves = tcurves[tokeep]
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
adn = peaks.loc[adn].sort_values().index.values
lmn = peaks.loc[lmn].sort_values().index.values

bin_size = 50
bins = np.arange(sws_ep['start'].iloc[0], sws_ep['end'].iloc[-1] + bin_size*1000, bin_size*1000)
rate = []
idx_spk = {}
for i,n in enumerate(lmn):
	count, _ = np.histogram(spikes[n].index.values, bins)
	rate.append(count)
	idx_spk[n] = nts.Tsd(t = spikes[n].index.values, d = np.digitize(spikes[n].index.values, bins))
	
rate = np.array(rate).T

newspikes = {n:[] for n in lmn}
# Pure WTA version
# idx = np.where((rate==np.vstack(rate.max(1))).sum(1) == 1)[0] + 1 #wta
# idx_n = lmn[np.argmax(rate[idx],1)]
# for n in lmn:
# 	t = idx_spk[n][idx_spk[n].as_series().isin(idx_bin)].index.values
# 	newspikes[n] = nts.Ts(t = t)
for i,n in enumerate(lmn):
	idx = np.where(rate[:,i] > 2)[0]+1
 	t = idx_spk[n][idx_spk[n].as_series().isin(idx)].index.values
 	newspikes[n] = nts.Ts(t = t)



cc = compute_CrossCorrs(newspikes, sws_ep, 2, 2000, norm=True)
cc = cc.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)
pairs = pd.Series(index = cc.columns, data = np.nan)
for i,j in pairs.index:	
	a = peaks[i] - peaks[j]
	pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))
pairs = pairs.dropna().sort_values()
cc = cc[pairs.index]

figure()
ax = subplot(211)
for i,n in enumerate(adn):
	tmp = spikes[n].restrict(sws_ep).fillna(peaks[n])
	plot(tmp, '|', markersize = 10)	
subplot(212, sharex = ax)
for i,n in enumerate(lmn):
	tmp = spikes[n].restrict(sws_ep).fillna(peaks[n]).as_series()
	plot(tmp, '|', markersize = 10)	
	tmp2 = newspikes[n].restrict(sws_ep).fillna(peaks[n]).as_series()
	plot(tmp2, 'o', markersize = 4)	

figure()
tmp = cc.T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 8), aspect = 'auto')
xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])


figure()
ax = subplot(211)
for i,n in enumerate(adn):
	tmp = spikes[n].restrict(sws_ep).fillna(peaks[n])
	plot(tmp, '|', markersize = 10)	
subplot(212, sharex = ax)
for i,n in enumerate(lmn):
	tmp2 = newspikes[n].restrict(sws_ep).fillna(peaks[n]).as_series()
	plot(tmp2, '|', markersize = 10)	

spikes2 = newspikes.copy()
spikes2.update({n:spikes[n] for n in adn})

from itertools import product

cc2 = []
for p in product(adn, lmn):
	cc2.append(compute_PairCrossCorr(spikes2, sws_ep, p, 2, 2000, True))
cc2 = pd.concat(cc2, 1)	
cc2.columns = list(product(adn, lmn))
pairs2 = pd.Series(index = cc2.columns, data = np.nan)
for i,j in pairs2.index:	
	a = peaks[i] - peaks[j]
	pairs2[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))
pairs2 = pairs2.dropna().sort_values()
cc2 = cc2[pairs2.index]

cc3 = []
for p in product(adn, lmn):
	cc3.append(compute_PairCrossCorr(spikes, sws_ep, p, 2, 2000, True))
cc3 = pd.concat(cc3, 1)	
cc3.columns = list(product(adn, lmn))
cc3 = cc3[pairs2.index]


figure()
subplot(121)
tmp = cc2.T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 8), aspect = 'auto')
xticks([0, np.where(cc2.index.values == 0)[0][0], len(cc2)], [cc2.index[0], 0, cc2.index[-1]])
subplot(122)
tmp = cc3.T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 8), aspect = 'auto')
xticks([0, np.where(cc3.index.values == 0)[0][0], len(cc3)], [cc3.index[0], 0, cc3.index[-1]])

figure()
for i in range(100):
	subplot(10,10,i+1)
	plot(cc2.iloc[:,i])
	plot(cc3.iloc[:,i])

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
from mpl_toolkits.mplot3d import Axes3D

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



tcurves = tcurves[tokeep]
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
adn = np.intersect1d(tokeep, np.where(shank<=3)[0])
lmn = np.intersect1d(tokeep, np.where(shank==5)[0])
adn = peaks.loc[adn].sort_values().index.values
lmn = peaks.loc[lmn].sort_values().index.values


speed = computeSpeed(position[['x', 'z']], wake_ep)
speed = speed.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)
idx = np.diff((speed > 0.005)*1.0)
start = np.where(idx == 1)[0]
end = np.where(idx == -1)[0]
if start[0] > end[0]:
	start = np.hstack(([0], start))
if start[-1] > end[-1]:
	end = np.hstack((end, [len(idx)]))

newwake_ep = nts.IntervalSet(start = speed.index.values[start], end = speed.index.values[end])
newwake_ep = newwake_ep.drop_short_intervals(1, time_units='s')

#######################################################################################
# PCA
#######################################################################################
wak_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, newwake_ep, 300, 3))
# wak_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, newwake_ep, 300, 3)
# wak_rate = wak_rate.values

sws_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, sws_ep, 50, 5)
sws_index = sws_rate.index.values
sws_rate = zscore_rate(sws_rate)
# sws_rate = sws_rate.values

wak_rate_adn = wak_rate[:,0:len(adn)]
wak_rate_lmn = wak_rate[:,-len(lmn):]

sws_rate_adn = sws_rate[:,0:len(adn)]
sws_rate_lmn = sws_rate[:,-len(lmn):]


# p_lmn = PCA(n_components=3).fit_transform(np.vstack((wak_rate,sws_rate)))
# ylmn = np.hstack((np.zeros(len(wak_rate_lmn)),np.ones(len(sws_rate_lmn))))
# scatter(p_lmn[0:len(wak_rate),0], p_lmn[0:len(wak_rate),1])

# pc_adn = PCA(n_components=2).fit(np.vstack((wak_rate_adn,sws_rate_adn)))
# pc_lmn = PCA(n_components=2).fit(np.vstack((wak_rate_lmn,sws_rate_lmn)))

pc_adn = PCA(n_components=3).fit(wak_rate_adn)
pc_lmn = PCA(n_components=3).fit(wak_rate_lmn)


p_wak_adn = pc_adn.transform(wak_rate_adn)
p_wak_lmn = pc_lmn.transform(wak_rate_lmn)

p_sws_adn = pc_adn.transform(sws_rate_adn)
p_sws_lmn = pc_lmn.transform(sws_rate_lmn)


angle = (np.pi/2) - np.arccos(p_sws_adn[:,2]/np.sqrt(np.sum(np.power(p_sws_adn,2),1)))
# angle = (np.pi/2) - np.arccos(p_sws_lmn[:,2]/np.sqrt(np.sum(np.power(p_sws_lmn,2),1)))
angle = np.abs(angle)
angle = pd.Series(index = sws_index, data = angle/(np.pi/2))
angle2 = angle.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)

# radius = pd.Series(index = sws_index, data = np.sqrt(np.sum(np.power(p_sws_lmn,2),1)))
# radius2 = radius.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
# radius2 -= radius2.min()
# radius2 /= radius2.max()

# threshold = 0.25
threshold = np.percentile(angle, 50)

index = np.diff(((angle2>threshold).values)*1.0)
#index = np.diff(((radius2>threshold).values)*1.0)
start = np.where(index == 1)[0]
end = np.where(index == -1)[0]
if start[0] > end[0]: end = end[1:]
if start[-1] > end[-1]: start = start[:-1]
good_ep = nts.IntervalSet(start = angle2.index.values[start], end = angle2.index.values[end])
good_ep = good_ep.intersect(sws_ep)
bad_ep = sws_ep.set_diff(good_ep)
bad_ep = bad_ep.drop_short_intervals(0)



fig = figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(p_wak_lmn[:,0], p_wak_lmn[:,1], p_wak_lmn[:,2])
ax.scatter(p_sws_lmn[0:10000,0], p_sws_lmn[0:10000,1], p_sws_lmn[0:10000,2])


figure()
ax = subplot(311)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(good_ep).fillna(peaks[n]), '|', markersize = 10, color = 'red')	
	plot(spikes[n].restrict(bad_ep).fillna(peaks[n]), '|', markersize = 10, color = 'blue')
subplot(312, sharex = ax)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(good_ep).fillna(peaks[n]), '|', markersize = 10, color = 'red')
	plot(spikes[n].restrict(bad_ep).fillna(peaks[n]), '|', markersize = 10, color = 'blue')	
subplot(313, sharex = ax)
plot(angle)
plot(angle2)
# plot(tmp2)
# plot(radius2)


#######################################################################################
# CROSS CORR
cc_good = compute_CrossCorrs({n:spikes[n] for n in lmn}, good_ep, 2, 3000, norm=True)
cc_bad = compute_CrossCorrs({n:spikes[n] for n in lmn}, bad_ep, 2, 3000, norm=True)




pairs = pd.Series(index = cc_good.columns, data = np.nan)
for i,j in pairs.index:	
	a = peaks[i] - peaks[j]
	pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))


pairs = pairs.dropna().sort_values()

cc_good = cc_good[pairs.index]
cc_bad = cc_bad[pairs.index]





figure()
subplot(221)
tmp = cc_good.T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 4), aspect = 'auto')
xticks([0, np.where(cc_good.index.values == 0)[0][0], len(cc_good)], [cc_good.index[0], 0, cc_good.index[-1]])
subplot(222)
tmp = cc_bad.T.values
imshow(scipy.ndimage.gaussian_filter(tmp, 4), aspect = 'auto')
xticks([0, np.where(cc_bad.index.values == 0)[0][0], len(cc_bad)], [cc_bad.index[0], 0, cc_bad.index[-1]])
subplot(223)
tmp = np.vstack((cc_good.loc[:-100].values,cc_good.loc[100:].values)).T
imshow(scipy.ndimage.gaussian_filter(tmp, 4), aspect = 'auto')
# xticks([0, np.where(cc_good.index.values == 0)[0][0], len(cc_good)], [cc_good.index[0], 0, cc_good.index[-1]])
subplot(224)
tmp = np.vstack((cc_bad.loc[:-100].values,cc_bad.loc[100:].values)).T
imshow(scipy.ndimage.gaussian_filter(tmp, 4), aspect = 'auto')
# xticks([0, np.where(cc_bad.index.values == 0)[0][0], len(cc_bad)], [cc_bad.index[0], 0, cc_bad.index[-1]])


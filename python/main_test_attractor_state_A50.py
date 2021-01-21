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


adn = np.intersect1d(tokeep, np.where(shank<=3)[0])
lmn = np.intersect1d(tokeep, np.where(shank==5)[0])

wak_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, newwake_ep, 300, 3))
wak_shuffle_rate = zscore_rate(binSpikeTrain(shuffleByIntervalSpikes({n:spikes[n] for n in tokeep}, newwake_ep), None, 300, 3))

sws_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, sws_ep, 5, 3)
index = sws_rate.index.values
sws_rate = zscore_rate(sws_rate)

Xtrain = np.vstack((wak_rate, wak_shuffle_rate))
Ytrain = np.hstack((np.zeros(len(wak_rate)), np.ones(len(wak_shuffle_rate))))

Xtrain_adn = Xtrain[:,0:len(adn)]
Xtrain_lmn = Xtrain[:,-len(lmn):]

kf = KFold(n_splits=3, shuffle=True)
c = np.zeros((len(Ytrain), 2))
for train_index, test_index in kf.split(Xtrain):
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0).fit(Xtrain_adn[train_index], Ytrain[train_index])
	c[test_index,0] = clf.predict(Xtrain_adn[test_index])	
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0).fit(Xtrain_lmn[train_index], Ytrain[train_index])
	c[test_index,1] = clf.predict(Xtrain_lmn[test_index])	


pc_adn = PCA(n_components=2).fit_transform(Xtrain_adn)
pc_lmn = PCA(n_components=2).fit_transform(Xtrain_lmn)

figure()
subplot(221)
scatter(pc_adn[:,0], pc_adn[:,1], c = Ytrain, s= 1)
title('adn')
subplot(222)
scatter(pc_adn[:,0], pc_adn[:,1], c = c[:,0], s= 1)
title('adn')
subplot(223)
scatter(pc_lmn[:,0], pc_lmn[:,1], c = Ytrain, s= 1)
title('lmn')
subplot(224)
scatter(pc_lmn[:,0], pc_lmn[:,1], c = c[:,1], s= 1)
title('lmn')



Xtest = sws_rate
p = np.zeros((len(Xtest),2))

clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
    max_depth=1, random_state=0).fit(Xtrain_adn, c[:,0])
p[:,0] = (np.diff(clf.predict_proba(Xtest[:,0:len(adn)]), 1)).flatten()
clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
    max_depth=1, random_state=0).fit(Xtrain_lmn, c[:,1])
p[:,1] = np.diff(clf.predict_proba(Xtest[:,-len(lmn):]), 1).flatten()


proba = nts.TsdFrame(t = index, d = p)



tcurves = tcurves[tokeep]
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
adn = peaks.loc[adn].sort_values().index.values
lmn = peaks.loc[lmn].sort_values().index.values


figure()
ax = subplot(211)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
subplot(212, sharex = ax)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
# subplot(313, sharex = ax)
# tmp = proba.as_dataframe()
# tmp2 = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
# plot(tmp)
# plot(tmp2)


figure()
ax = subplot(211)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(wake_ep).fillna(peaks[n]), '|', markersize = 10)	
subplot(212, sharex = ax)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(wake_ep).fillna(peaks[n]), '|', markersize = 10)	


sys.exit()

##################################
# TEST removing some spikes
##################################
figure()
ax = subplot(211)
for i,n in enumerate(adn):
	tmp = spikes[n].restrict(sws_ep).fillna(peaks[n])
	plot(tmp, '|', markersize = 10)	
subplot(212, sharex = ax)
for i,n in enumerate(lmn):
	tmp = spikes[n].restrict(sws_ep).fillna(peaks[n]).as_series().sample(frac=0.5)
	plot(tmp, '|', markersize = 10)	





threshold = 0.25
index = np.diff(((tmp2>threshold).values)*1.0)
start = np.where(index == 1)[0]
end = np.where(index == -1)[0]
good_ep = nts.IntervalSet(start = tmp2.index.values[start], end = tmp2.index.values[end])

bad_ep = sws_ep.set_diff(good_ep)


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
tmp = np.vstack((cc_good.loc[:-20].values,cc_good.loc[20:].values)).T
imshow(scipy.ndimage.gaussian_filter(tmp, 4), aspect = 'auto')
# xticks([0, np.where(cc_good.index.values == 0)[0][0], len(cc_good)], [cc_good.index[0], 0, cc_good.index[-1]])
subplot(224)
tmp = np.vstack((cc_bad.loc[:-20].values,cc_bad.loc[20:].values)).T
imshow(scipy.ndimage.gaussian_filter(tmp, 4), aspect = 'auto')
# xticks([0, np.where(cc_bad.index.values == 0)[0][0], len(cc_bad)], [cc_bad.index[0], 0, cc_bad.index[-1]])

# cc_wak = compute_CrossCorrs({n:spikes[n] for n in lmn}, wake_ep, norm=True)
# cc_wak = cc_wak[pairs.index]
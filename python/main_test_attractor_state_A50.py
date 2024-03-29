import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
newwake_ep = newwake_ep.drop_short_intervals(10, time_units='s')


adn = np.intersect1d(tokeep, np.where(shank<=3)[0])
lmn = np.intersect1d(tokeep, np.where(shank==5)[0])

#########################################################
# BINNING SPIKE TRAIN
#########################################################
wak_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, newwake_ep, 300, 3)
wak_rate = wak_rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis=0).mean(std=2.0)
wak_rate = zscore_rate(wak_rate)

shspike = shuffleByOrderSpikes({n:spikes[n] for n in tokeep}, newwake_ep) # made on one epoch only
wak_shuffle_rate = binSpikeTrain(shspike, None, 300, 3)
wak_shuffle_rate = wak_shuffle_rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis=0).mean(std=2.0)
wak_shuffle_rate = zscore_rate(wak_shuffle_rate)

Xtrain = np.vstack((wak_rate, wak_shuffle_rate))
Ytrain = np.hstack((np.zeros(len(wak_rate)), np.ones(len(wak_shuffle_rate))))

Xtrain_adn = Xtrain[:,0:len(adn)]
Xtrain_lmn = Xtrain[:,-len(lmn):]

kf = KFold(n_splits=5, shuffle=True)
c = np.zeros((len(Ytrain), 2))
for train_index, test_index in kf.split(Xtrain):
	clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01,
    max_depth=3, random_state=0,verbose=True, tol=1e-10).fit(Xtrain_adn[train_index], Ytrain[train_index])
	c[test_index,0] = clf.predict(Xtrain_adn[test_index])	
	clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01,
    max_depth=3, random_state=0,verbose=True, tol=1e-10).fit(Xtrain_lmn[train_index], Ytrain[train_index])
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

show()

sys.exit()

sws_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, sws_ep, 10, 2)
#sws_rate = sws_rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis=0).mean(std=1.0)
index = sws_rate.index.values
sws_rate = zscore_rate(sws_rate)

Xtest = sws_rate
p = np.zeros((len(Xtest),2))

Xtest_adn = Xtest[:,0:len(adn)]
Xtest_lmn = Xtest[:,-len(lmn):]


clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01,
    max_depth=3, random_state=0,verbose=True, tol=1e-10).fit(Xtrain_adn, Ytrain)
p[:,0] = clf.predict_proba(Xtest_adn)[:,0]
clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01,
    max_depth=3, random_state=0, verbose = True, tol=1e-10).fit(Xtrain_lmn, Ytrain)
p[:,1] = clf.predict_proba(Xtest_lmn)[:,0]


proba = nts.TsdFrame(t = index, d = p)

tcurves = tcurves[tokeep]
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
adn = peaks.loc[adn].sort_values().index.values
lmn = peaks.loc[lmn].sort_values().index.values

tmp = proba.as_dataframe()
tmp2 = 1/(1+np.exp(-20*(tmp - 0.5)))
tmp2 = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
#plot(tmp)
#plot(tmp2)

ex_sws = nts.IntervalSet(start = 4399305437.713542, end = 4403054216.186978)

figure()
ax = subplot(211)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
ax2 = ax.twinx()
plot(tmp[0], alpha = 0.5)
plot(tmp2[0])
xlim(ex_sws.loc[0,'start'], ex_sws.loc[0,'end'])

ax = subplot(212)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
ax2 = ax.twinx()
plot(tmp[1], alpha = 0.5)
plot(tmp2[1])
xlim(ex_sws.loc[0,'start'], ex_sws.loc[0,'end'])

figure()
subplot(211)
hist(tmp[0], 50, label = 'ADN')
subplot(212)
hist(tmp[1], 50, label = 'LMN')
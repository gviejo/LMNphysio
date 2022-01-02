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
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	return rate

def softmax(r, m, b):
	return 1/(1+np.exp(-b*(r - m)))

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

down_ep, up_ep 						= loadUpDown(data_directory)

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
wak_rate = zscore_rate(wak_rate)
#wak_rate = softmax(wak_rate, wak_rate.mean(), 1)

#sys.exit()
# pc_adn = PCA(n_components=2).fit_transform(wak_rate[:,0:len(adn)].T)
# pc_lmn = PCA(n_components=2).fit_transform(wak_rate[:,-len(adn):].T)

sws_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, sws_ep, 15, 3)
sws_rate = sws_rate[sws_rate.sum(1)>4]
sws_rate = zscore_rate(sws_rate)
#sws_rate = softmax(sws_rate, sws_rate.mean(), 2)

wak_rate = wak_rate.values
sws_rate = sws_rate.values

corr_adn = np.corrcoef(wak_rate[:,0:len(adn)].T)
corr_lmn = np.corrcoef(wak_rate[:,-len(lmn):].T)
corr_adn[np.diag_indices_from(corr_adn)] = 0
corr_lmn[np.diag_indices_from(corr_lmn)] = 0

r_adn_sws = np.sum(np.dot(sws_rate[:,0:len(adn)], corr_adn)*sws_rate[:,0:len(adn)], 1)
r_lmn_sws = np.sum(np.dot(sws_rate[:,-len(lmn):], corr_lmn)*sws_rate[:,-len(lmn):], 1)
r_sws = np.vstack((r_adn_sws, r_lmn_sws)).T
r_adn_wak = np.sum(np.dot(wak_rate[:,0:len(adn)], corr_adn)*wak_rate[:,0:len(adn)], 1)
r_lmn_wak = np.sum(np.dot(wak_rate[:,-len(lmn):], corr_lmn)*wak_rate[:,-len(lmn):], 1)
r_wak = np.vstack((r_adn_wak, r_lmn_wak)).T

corr_state = {'wak':[], 'sws':[]}
# State correlation shifted in time
for i in range(-20, 20):
	for e, r in zip(['wak', 'sws'], [r_wak, r_sws]):
		corr_state[e].append(scipy.stats.pearsonr(r[np.maximum(-i,0):np.minimum(len(r),len(r)-i),1], 
			r[np.maximum(i,0):np.minimum(len(r),len(r)+i),0])[0])

corr_state = pd.DataFrame(corr_state)

figure()
plot(corr_state['wak'], label = 'wake')
plot(corr_state['sws'], label = 'sws')
legend()
show()


sys.exit()

rea = pd.DataFrame(index = index, data = np.vstack([r_adn, r_lmn]).T)


tcurves = tcurves[tokeep]
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
adn = peaks.loc[adn].sort_values().index.values
lmn = peaks.loc[lmn].sort_values().index.values

tmp = rea
#tmp2 = 1/(1+np.exp(-20*(tmp - 0.5)))
tmp2 = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
#plot(tmp)
#plot(tmp2)

ex_sws = nts.IntervalSet(start = 4399305437.713542, end = 4403054216.186978)

figure()
ax = subplot(211)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
ax2 = ax.twinx()
plot(tmp[0])
#plot(tmp2[0])
xlim(ex_sws.loc[0,'start'], ex_sws.loc[0,'end'])

ax = subplot(212)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
ax2 = ax.twinx()
plot(tmp[1])
#plot(tmp2[1])
xlim(ex_sws.loc[0,'start'], ex_sws.loc[0,'end'])

show()


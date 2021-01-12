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
import matplotlib.gridspec as gridspec

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


##################################################################
# Compute Cross-corr by removing some spikes
##################################################################
ccs = {'adn':{}, 'lmn':{}}

tocut = np.arange(0, 1, 0.1)

for i, cut in enumerate(tocut):
	for gr, name in zip([adn, lmn], ['adn', 'lmn']):
		tmp = {n:nts.Ts(spikes[n].restrict(sws_ep).as_series().sample(frac=1-cut)) for n in gr}
		cc = compute_CrossCorrs(tmp, sws_ep, 2, 2000, norm=True)
		cc = cc.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)

		tcurves = tuning_curves[gr]
		peaks 	= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()
		tcurves 		= tcurves[peaks.index.values]

		new_index = cc.columns
		pairs = pd.Series(index = new_index, data = np.nan)
		for i,j in pairs.index:	
			a = peaks[i] - peaks[j]
			pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))
		pairs = pairs.dropna().sort_values()

		cc = cc[pairs.index]
		ccs[name][cut] = cc


figure()
gs = gridspec.GridSpec(2, len(tocut))
for i, cut in enumerate(tocut):
	for j, name in enumerate(['adn', 'lmn']):
		subplot(gs[j,i])
		tmp = ccs[name][cut].T.values
		imshow(scipy.ndimage.gaussian_filter(tmp, 2), aspect = 'auto', cmap = 'jet')		



sys.exit()


tcurves = tcurves[tokeep]
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
adn = peaks.loc[adn].sort_values().index.values
lmn = peaks.loc[lmn].sort_values().index.values


figure()
ax = subplot(311)
for i,n in enumerate(adn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
subplot(312, sharex = ax)
for i,n in enumerate(lmn):
	plot(spikes[n].restrict(sws_ep).fillna(peaks[n]), '|', markersize = 10)	
subplot(313, sharex = ax)
tmp = proba.as_dataframe()
tmp2 = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
plot(tmp)
# plot(tmp2)

sys.exit()



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
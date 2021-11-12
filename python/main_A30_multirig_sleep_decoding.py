import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pingouin import partial_corr
from umap import UMAP
import sys
from matplotlib.colors import hsv_to_rgb
import hsluv
from sklearn.manifold import Isomap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

def zscore_rate(rate):
	idx = rate.index
	cols = rate.columns
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	rate = pd.DataFrame(index = idx, data = rate, columns = cols)
	return nts.TsdFrame(rate)


data_directory = '/mnt/Data2/LMN-PSB-2/A3013/A3013-210806A'

#episodes = ['sleep', 'wake']
episodes = ['sleep', 'wake', 'wake', 'sleep', 'wake', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']

events = ['1', '2', '4', '5']




spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)


position 							= loadPosition(data_directory, events, episodes, 2, 1)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
sws_ep								= loadEpoch(data_directory, 'sws')
rem_ep 								= loadEpoch(data_directory, 'rem')

#################
# TUNING CURVES
tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60)
#tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tuning_curves 						= smoothAngularTuningCurves(tuning_curves, 10, 2)

tokeep, stat 						= findHDCells(tuning_curves, z=1, p = 0.001)


###################################################################################################
neurons = [3,4,5,7,10,12,16,18,20,21,23,24,26,27,29,31,32,35,37,38,39,41,50,52,59,62,63]

###################################################################################################
# TRAINING DATA
###################################################################################################
bin_size = 300

data = []
sessions = []

for e in [0,3]:
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

	# cutting the 20th percentile
	tmp = rate.values
	index = tmp.sum(1) > np.percentile(tmp.sum(1), 20)

	tmp = tmp[index,:]

	data.append(tmp)
	sessions.append(np.ones(len(tmp))*e)

data = np.vstack(data)
sessions = np.hstack(sessions)

###################################################################################################
# TEST DATA
###################################################################################################


testdata = []
for e in [0,2]:
	nep = rem_ep.intersect(sleep_ep.loc[[e]])

	bin_size = 100

	ep = sleep_ep.loc[[e]]

	bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)

	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
	for i in neurons:
		spks = spikes[i].as_units('ms').index.values
		spike_counts[i], _ = np.histogram(spks, bins)

	rate = np.sqrt(spike_counts/(bin_size*1e-3))
	rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
	rate = nts.TsdFrame(t = rate.index.values, d = rate.values, time_units = 'ms')
	rate = rate.restrict(nep)

	testdata.append(rate)



###################################################################################################
# SCIKIT-learn
###################################################################################################
y = sessions.copy()
for i, n in enumerate(np.unique(sessions)):
	y[sessions == n] = i

X = data

Xt1 = testdata[1].values
Xt0 = testdata[0].values

clf = LogisticRegression(random_state=0, tol = 1e-6).fit(X, y)

proba1 = clf.predict_proba(Xt1)
proba0 = clf.predict_proba(Xt0)

proba1 = pd.DataFrame(index = testdata[1].index.values, data = proba1)
proba0 = pd.DataFrame(index = testdata[0].index.values, data = proba0)

figure()
subplot(211)
hist(proba0[0], 200)
subplot(212)
hist(proba1[0], 200)
show()

figure()
ax = subplot(211)
plot(proba1[1])

subplot(212, sharex = ax)
plot(testdata[1].sum(1))

show()


sys.exit()
###################################################################################################
# XGBOOST
###################################################################################################
clas = sessions.copy()
for i, n in enumerate(np.unique(sessions)):
	clas[sessions == n] = i

data = pd.DataFrame(data)
clas = pd.DataFrame(data = clas, columns = ['classe'])
datatest = pd.DataFrame(datatest.values)

dtrain = xgb.DMatrix(data, label=clas)
dtest = xgb.DMatrix(datatest)


param = {	'max_depth':5, 
			'eta':1, 
			'objective':'binary:logistic',
			'learning_rate': 0.05,			
			'gamma':0.5			
		}
num_round = 100
bst = xgb.train(param, dtrain, num_round, verbose_eval = 4)
# make prediction
preds = bst.predict(dtest)


figure()
ax = subplot(211)
plot(preds)

subplot(212, sharex = ax)
plot(datatest.sum(1))

show()
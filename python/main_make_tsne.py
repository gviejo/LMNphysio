import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys, os
from sklearn.manifold import TSNE

data_directory = '../data/A1400/A1407/'

info = pd.read_csv(data_directory+'A1407.csv')
info = info.set_index('Session')

sessions = os.listdir(data_directory)
sessions.remove('A1407.csv') 
sessions = np.sort(sessions)


wakeall = []
sleepall = []
frate = []

for s in sessions:
	path 								= os.path.join(data_directory, s)
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	episodes 							= info.filter(like='Trial').loc[s].dropna().values
	events								= list(np.where(episodes == 'wake')[0].astype('str'))
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sleep_ep 							= loadEpoch(path, 'sleep')					
	acceleration						= loadAuxiliary(path)
	sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)
	autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep, 2, 200)
	autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep, 2, 200)
	mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, sleep_ep], ['wake', 'sleep'])
	index 								= [s+'_'+str(k) for k in spikes]
	autocorr_wake.columns 				= pd.Index(index)
	autocorr_sleep.columns 				= pd.Index(index)
	mean_frate.index 					= pd.Index(index)
	wakeall.append(autocorr_wake)
	sleepall.append(autocorr_sleep)
	frate.append(mean_frate)


sleepall = pd.concat(sleepall, 1)
wakeall = pd.concat(wakeall, 1)

# 1. starting at 2
autocorr_wak = wakeall.loc[0.5:]
autocorr_sle = sleepall.loc[0.5:]

# 3. lower than 200 
autocorr_wak = autocorr_wak.drop(autocorr_wak.columns[autocorr_wak.apply(lambda col: col.max() > 100.0)], axis = 1)
autocorr_sle = autocorr_sle.drop(autocorr_sle.columns[autocorr_sle.apply(lambda col: col.max() > 100.0)], axis = 1)

# # 4. gauss filt
autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
autocorr_sle = autocorr_sle.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)

autocorr_wak = autocorr_wak[2:200]
autocorr_sle = autocorr_sle[2:200]


# 6 combining all 
neurons = np.intersect1d(autocorr_wak.columns, autocorr_sle.columns)
# neurons = np.intersect1d(neurons, fr_index)

data = np.hstack([autocorr_wak[neurons].values.T,autocorr_sle[neurons].values.T])


from sklearn.cluster import KMeans



K = KMeans(n_clusters = 4, random_state = 0).fit(data).labels_

X = TSNE(2, 100).fit_transform(data)


scatter(X[:,0], X[:,1], c = K)

show()

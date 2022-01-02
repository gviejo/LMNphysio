import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from itertools import combinations
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
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ripples.list'), delimiter = '\n', dtype = str, comments = '#')
lmnshank = pd.read_csv(os.path.join(data_directory,'LMN_shanks.txt'), header = None, index_col = 0)

infos = getAllInfos(data_directory, datasets)

allcc = []

for s in datasets:
	print(s)
	name = s.split('/')[-1]
	path = os.path.join(data_directory, s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	episodes 							= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	episodes[episodes != 'sleep'] 		= 'wake'
	events								= list(np.where(episodes != 'sleep')[0].astype('str'))	
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sws_ep								= loadEpoch(path, 'sws')
	rip_ep, rip_tsd 					= loadRipples(os.path.join(data_directory, s))

	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################	
	# selecting lmn neurons
	neurons = np.where(np.sum([shank==i for i in np.fromstring(lmnshank.iloc[1].values[0], dtype=int, sep=' ')], 0))[0]

	tuning_curves = computeAngularTuningCurves({n:spikes[n] for n in neurons}, position['ry'], wake_ep, 121)	
	tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

	# CHECKING HALF EPOCHS
	wake2_ep = splitWake(wake_ep)
	tokeep2 = []
	stats2 = []
	tcurves2 = []
	for i in range(2):
		# tcurves_half = computeLMNAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])[0][1]
		tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
		tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)
		tokeep, stat = findHDCells(tcurves_half)
		tokeep2.append(tokeep)
		stats2.append(stat)
		tcurves2.append(tcurves_half)

	tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

	if len(tokeep) >= 7:

		sys.exit()
		############################################################################################### 
		# Computing attractor reactivation
		###############################################################################################
		wak_rate = binSpikeTrain({n:spikes[n] for n in tokeep}, wake_ep, 300, 5)
		wak_rate = zscore_rate(wak_rate)

		shspike = shuffleByOrderSpikes({n:spikes[n] for n in tokeep}, wake_ep) # made on one epoch only
		wak_shuffle_rate = binSpikeTrain(shspike, None, 300, 5)
		wak_shuffle_rate = zscore_rate(wak_shuffle_rate)

		Xtrain = np.vstack((wak_rate, wak_shuffle_rate))
		Ytrain = np.hstack((np.zeros(len(wak_rate)), np.ones(len(wak_shuffle_rate))))


		kf = KFold(n_splits=5, shuffle=True)
		c = np.zeros(len(Ytrain))
		for train_index, test_index in kf.split(Xtrain):
			clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01,
		    max_depth=3, random_state=0,verbose=True, tol=1e-10).fit(Xtrain[train_index], Ytrain[train_index])
			c[test_index] = clf.predict(Xtrain[test_index])	


		pc = PCA(n_components=2).fit_transform(Xtrain)		

		figure()
		subplot(121)
		scatter(pc[:,0], pc[:,1], c = Ytrain, s= 1)		
		subplot(222)
		scatter(pc[:,0], pc[:,1], c = c, s= 1)

		show()


		


		############################################################################################### 
		# SWR cross-corr
		###############################################################################################

		cc_rip = compute_EventCrossCorr({n:spikes[n] for n in tokeep}, rip_tsd, sws_ep, binsize = 10, nbins = 400, norm=True)

		cc_rip.columns = [name + '_' + str(n) for n in tokeep]
		#######################
		# SAVING
		#######################
		allcc.append(cc_rip)
	

allcc = pd.concat(allcc, 1)

figure()
plot(allcc.mean(1))
show()


datatosave = {'allcc':allcc}
cPickle.dump(datatosave, open(os.path.join('../data/', 'LMN_SWR_CC.pickle'), 'wb'))


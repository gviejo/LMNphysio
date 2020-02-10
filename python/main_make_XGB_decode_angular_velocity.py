import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold



def extract_tree_threshold(trees):
    """ Take BST TREE and return a dict = {features index : [splits position 1, splits position 2, ...]}
    """
    n = len(trees.get_dump())
    thr = {}
    for t in range(n):
        gv = xgb.to_graphviz(trees, num_trees=t)
        body = gv.body		
        for i in range(len(body)):
            for l in body[i].split('"'):
                if 'f' in l and '<' in l:
                    tmp = l.split("<")
                    if tmp[0] in thr:
                        thr[tmp[0]].append(float(tmp[1]))
                    else:
                        thr[tmp[0]] = [float(tmp[1])]
    for k in thr:
        thr[k] = np.sort(np.array(thr[k]))
    return thr

def xgb_decodage(Xr, Yr, Xt, n_class):          
	dtrain = xgb.DMatrix(Xr, label=Yr)
	dtest = xgb.DMatrix(Xt)

	params = {'objective': "multi:softprob",
	'eval_metric': "mlogloss", #loglikelihood loss
	'seed': np.random.randint(1, 10000), #for reproducibility
	'silent': 1,
	'learning_rate': 0.01,
	'min_child_weight': 2, 
	'n_estimators': 100,
	# 'subsample': 0.5,
	'max_depth': 5, 
	'gamma': 0.5,
	'num_class':n_class}

	num_round = 2000
	bst = xgb.train(params, dtrain, num_round)
	ymat = bst.predict(dtest)
	pclas = np.argmax(ymat, 1)
	return pclas

def fit_cv(X, Y, n_class, n_cv=10, verbose=1, shuffle = False):
	if np.ndim(X)==1:
		X = np.transpose(np.atleast_2d(X))
	cv_kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
	skf  = cv_kf.split(X)    
	Y_hat=np.zeros(len(Y))*np.nan	

	for idx_r, idx_t in skf:        
		Xr = np.copy(X[idx_r, :])
		Yr = np.copy(Y[idx_r])
		Xt = np.copy(X[idx_t, :])
		Yt = np.copy(Y[idx_t])
		if shuffle: np.random.shuffle(Yr)
		Yt_hat = xgb_decodage(Xr, Yr, Xt, n_class)
		Y_hat[idx_t] = Yt_hat
		
	return Y_hat


data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

sessions = ['A1407-190411', 'A1407-190416', 'A1407-190417', 'A1407-190422']

for s in sessions:
	path = os.path.join(data_directory, s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	episodes 							= info.filter(like='Trial').loc[s].dropna().values
	events								= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sleep_ep 							= loadEpoch(path, 'sleep')					
	# sws_ep								= loadEpoch(path, 'sws')
	# rem_ep								= loadEpoch(path, 'rem')

	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################
	# tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 121)

	# for i in tuning_curves:
	# 	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)

	# tokeep, stat 						= findHDCells(tuning_curves[1])

	# neurons = tokeep

	neurons = [0, 9, 10, 12, 14]

	# neurons = list(spikes.keys())

	####################################################################################################################
	# MAKE VELOCITY
	####################################################################################################################
	bin_size 		= 10 # ms
	angle 			= position['ry']	
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))	
	tmp2 			= tmp.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=30.0)
	time_bins		= np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size*1000) # assuming microseconds
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp3 			= tmp2.groupby(index).mean()
	tmp3.index 		= time_bins[np.unique(index)-1]+bin_size/2 * 1000
	tmp3 			= nts.Tsd(tmp3)	
	tmp4			= np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
	velocity 		= nts.Tsd(t=tmp3.index.values[1:], d = tmp4)
	

	# ax = subplot(211)
	# plot(tmp3)
	# subplot(212, sharex = ax)
	# plot(velocity)

	####################################################################################################################
	# BIN WAKE
	####################################################################################################################	
	spike_counts = pd.DataFrame(index = time_bins[0:-1]+np.diff(time_bins)/2, columns = neurons)
	for i in neurons:
		spks = spikes[i].index.values
		spike_counts[i], _ = np.histogram(spks, time_bins)


	############################################################################################### 
	# DECODING ANGULAR VELOCITY WITH XGBBOST
	###############################################################################################
	X_spikes = 	spike_counts.values
	# keeping points between -pi and pi
	edges = np.linspace(-np.pi, np.pi, 5)
	index = np.digitize(velocity.values, edges)-1
	Y_velocity = index[np.logical_and(index >= 0, index <= len(edges)-2)]
	X_spikes = X_spikes[np.logical_and(index >= 0, index <= len(edges)-2)]


	Y_hat = fit_cv(X_spikes, Y_velocity, n_class = len(edges)-1, n_cv=5, verbose=1, shuffle = False)


	plot(Y_velocity)
	plot(Y_hat)


	sys.exit()

############################################################################################### 
# PLOT
###############################################################################################


figure()
for i in spikes:
	subplot(3,5,i+1, projection = 'polar')
	plot(tuning_curves[1][i], label = str(shank[i]))
	legend()
show()



figure()
subplot(121)
plot(velocity)
subplot(122)
hist(velocity, 1000)
[axvline(e) for e in edges[1:-1]]


figure()
style = ['--', '-', '--']
colors = ['black', 'red', 'black']
alphas = [0.7, 1, 0.7]
for i in spikes:
	subplot(6,7,i+1)
	for j in range(3):
	# for j in [1]:
		tmp = tuning_curves[j][i] #- mean_frate.loc[i,'wake']
		plot(tmp, linestyle = style[j], color = colors[j], alpha = alphas[j])
	title(str(shank[i]))



figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(autocorr_wake[i], label = str(shank[i]))
	plot(autocorr_sleep[i])
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(velo_curves[i], label = str(shank[i]))
	legend()

figure()
for i in spikes:
	subplot(6,7,i+1)
	imshow(spatial_curves[i])
	colorbar()

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(speed_curves[i], label = str(shank[i]))
	legend()

show()



import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from sklearn.preprocessing import StandardScaler
from pyglmnet import GLM


def compute_GLM_CrossCorrs(spks, ep, bin_size=10, lag=5000, lag_size=50, sigma = 15):
	bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
	order = list(spks.keys())
	time_lag = np.arange(-lag, lag, lag_size, dtype = np.int)

	shift = time_lag//bin_size

	# all spike counts
	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = order)
	for k in spike_counts:
		spike_counts[k] = np.histogram(spks[k].restrict(ep).as_units('ms').index.values, bins)[0]

	spike_counts = nts.TsdFrame(t = spike_counts.index.values, d = spike_counts.values, time_units = 'ms')
	spike_counts = spike_counts.restrict(ep)
	spike_counts = spike_counts.as_dataframe()
	glmcc = pd.DataFrame(index = time_lag, columns = list(combinations(order, 2)), dtype = np.float32)

	count = 0
	for pair in glmcc.columns:
		print(count/len(glmcc.columns))
		count += 1
		# computing predictors	
		X1 = spike_counts[pair[1]]	# predictor 
		X2 = spike_counts.drop(list(pair), axis = 1).sum(1) # population
		X1 = X1.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = sigma)
		X2 = X2.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 3)
		X_train = pd.concat([X1,X2], axis = 1)
		X_train = X_train - X_train.mean(0)
		X_train = X_train / X_train.std(0)

		# target at different time lag				
		b = []
		for t, s in zip(time_lag, shift):
			glm = GLM(distr="poisson", reg_lambda=0.1, solver = 'cdfast', score_metric = 'deviance', max_iter = 200)
			# y_train = np.histogram(spks[pair[0]].restrict(ep).as_units('ms').index.values+t, bins)[0]
			y_train = spike_counts[pair[0]].values			
			if s < 0 : # backward
				y_train = y_train[-s:]
				Xtrain = X_train.values[:s]
			elif s > 0 : # forward
				y_train = y_train[:-s]
				Xtrain = X_train.values[s:]
			else:
				Xtrain = X_train.values

			# sys.exit()
			# fit glm			
			glm.fit(Xtrain, y_train)
			b.append(np.float32(glm.beta_[0]))


		glmcc[pair] = np.array(b)

	return glmcc



data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

sessions = ['A1407-190416', 'A1407-190417', 'A1407-190422']

allcc_wak = []
allcc_rem = []
allcc_sws = []
allpairs = []
alltcurves = []
allfrates = []
allvcurves = []
allscurves = []
allpeaks = []

for s in sessions:
	print(s)
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
	sws_ep								= loadEpoch(path, 'sws')
	rem_ep								= loadEpoch(path, 'rem')

	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################
	tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
	spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 21)
	autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)
	autocorr_sleep, frate_sleep 		= compute_AutoCorrs(spikes, sleep_ep)
	velo_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, nb_bins = 30)
	mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])
	speed_curves 						= computeSpeedTuningCurves(spikes, position[['x', 'z']], wake_ep)


	for i in tuning_curves:
		tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)

	velo_curves = velo_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
	speed_curves = speed_curves.rolling(window=5, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)

			
	############################################################################################### 
	# GLM CROSS CORRELATION
	##############################################################################################	

	cc_wak = compute_GLM_CrossCorrs(spikes, wake_ep, bin_size=100, lag=2000, lag_size=10)

	cc_rem = compute_GLM_CrossCorrs(spikes, rem_ep, bin_size=100, lag=2000, lag_size=20)

	cc_sws = compute_GLM_CrossCorrs(spikes, sws_ep, bin_size=10, lag=200, lag_size=20, sigma = 1)




	# cc_wak = compute_CrossCorrs(spikes, wake_ep)
	# cc_rem = compute_CrossCorrs(spikes, rem_ep)
	# cc_sws = compute_CrossCorrs(spikes, sws_ep, 1, 200)

	# cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	# cc_rem = cc_rem.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	# cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)



	# sorting by angular differences
	tokeep, stat 						= findHDCells(tuning_curves[1])

	# if s == 'A1407-190416':
	# 	tokeep = np.delete(tokeep, np.where(tokeep==5))
	# 	tokeep = np.delete(tokeep, np.where(tokeep==2))


	tcurves 							= tuning_curves[1][tokeep]
	velo_curves 						= velo_curves[tokeep]
	speed_curves						= speed_curves[tokeep]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
	tcurves 							= tcurves[peaks.index.values]
	velo_curves							= velo_curves[peaks.index.values]
	speed_curves 						= speed_curves[peaks.index.values]
	mean_frate							= mean_frate.loc[peaks.index.values]
	neurons 							= [s+'_'+str(n) for n in tcurves.columns.values]
	peaks.index							= pd.Index(neurons)
	tcurves.columns						= pd.Index(neurons)
	velo_curves.columns 				= pd.Index(neurons)
	speed_curves.columns 				= pd.Index(neurons)
	mean_frate.index 					= pd.Index(neurons)
	

	new_index = [(s+'_'+str(i),s+'_'+str(j)) for i,j in cc_wak.columns]
	cc_wak.columns = pd.Index(new_index)
	cc_rem.columns = pd.Index(new_index)
	cc_sws.columns = pd.Index(new_index)
	pairs = pd.Series(index = new_index)
	for i,j in pairs.index:	
		if i in neurons and j in neurons:
			a = peaks[i] - peaks[j]
			pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))

	pairs = pairs.dropna().sort_values()

	#######################
	# SAVING
	#######################
	alltcurves.append(tcurves)
	allvcurves.append(velo_curves)
	allscurves.append(speed_curves)
	allpairs.append(pairs)
	allcc_wak.append(cc_wak[pairs.index])
	allcc_rem.append(cc_rem[pairs.index])
	allcc_sws.append(cc_sws[pairs.index])
	allfrates.append(mean_frate)
	allpeaks.append(peaks)

 
alltcurves 	= pd.concat(alltcurves, 1)
allscurves 	= pd.concat(allscurves, 1)
allvcurves 	= pd.concat(allvcurves, 1)
allpairs 	= pd.concat(allpairs, 0)
allfrates 	= pd.concat(allfrates, 0)
allcc_wak 	= pd.concat(allcc_wak, 1)
allcc_rem 	= pd.concat(allcc_rem, 1)
allcc_sws 	= pd.concat(allcc_sws, 1)
allpeaks 	= pd.concat(allpeaks, 0)


allpairs = allpairs.sort_values()
allfrates = allfrates.astype('float')

sess_groups = pd.DataFrame(pd.Series({k:k.split("_")[0] for k in alltcurves.columns.values})).groupby(0).groups

colors = ['blue', 'red', 'green']


datatosave = {	'tcurves':alltcurves,
				'sess_groups':sess_groups,
				'frates':allfrates,
				'cc_wak':allcc_wak,
				'cc_rem':allcc_rem,
				'cc_sws':allcc_sws,
				'pairs':allpairs,
				'peaks':allpeaks
				}

cPickle.dump(datatosave, open('../figures/figures_poster_2019/fig_2_crosscorr_glm.pickle', 'wb'))


##########################################################
# TUNING CURVES
figure()
for i, g in enumerate(sess_groups.keys()):
	for j, n in enumerate(sess_groups[g]):
		subplot(3,8,j+1+i*8, projection = 'polar')
		plot(alltcurves[n], color = colors[i])
		

##########################################################
# ANGULAR VELOCITY CURVEs
figure()
for i, g in enumerate(sess_groups.keys()):
	for j, n in enumerate(sess_groups[g]):
		subplot(3,8,j+1+i*8)
		plot(allvcurves[n], color = colors[i])


##########################################################
# SPEED CURVEs
figure()
for i, g in enumerate(sess_groups.keys()):
	for j, n in enumerate(sess_groups[g]):
		subplot(3,8,j+1+i*8)
		plot(allscurves[n], color = colors[i])

##########################################################
# CROSS CORR
titles = ['wake', 'REM', 'NREM']
figure()
subplot(221)
scatter(np.log(allfrates['wake'].values), np.log(allfrates['rem'].values))
xlabel("Wake")
ylabel("REM")
subplot(222)
scatter(np.log(allfrates['wake'].values), np.log(allfrates['sws'].values))
xlabel("Wake")
ylabel("SWS")
for i, cc in enumerate([allcc_wak, allcc_rem, allcc_sws]):
	subplot(2,3,i+1+3)
	tmp = cc[allpairs.index].values
	tmp = tmp - tmp.mean(0)
	tmp = tmp / tmp.std(0)
	imshow(tmp.T, aspect = 'auto', cmap = 'jet')
	title(titles[i])
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])



##########################################################
# EXEMPLES
groups = allpairs.groupby(np.digitize(allpairs, [0, np.pi/3, 2*np.pi/3, np.pi])).groups

figure()
for i, g in enumerate(groups.keys()):
	for j, cc in enumerate([allcc_wak, allcc_rem, allcc_sws]):
		subplot(3,3,j+1+i*3)
		plot(cc[groups[g]], color = 'grey', alpha = 0.6)
		if i == 0:
			title(titles[j])

show()








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



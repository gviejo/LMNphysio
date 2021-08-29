import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec

def zscore_rate(rate):
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	return rate



data_directory = '/mnt/Data2/Opto/A8000/A8015/A8015-210826A'

episodes = ['sleep', 'wake', 'sleep', 'wake']
# events = ['1', '3']

# episodes = ['sleep', 'wake', 'sleep']
events = ['1', '3']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)


position 							= loadPosition(data_directory, events, episodes, 2, 1)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
sws_ep								= loadEpoch(data_directory, 'sws')
acceleration						= loadAuxiliary(data_directory)
# #sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

#################
# TUNING CURVES
tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60)
#tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tuning_curves 						= smoothAngularTuningCurves(tuning_curves, 10, 2)

tokeep, stat 						= findHDCells(tuning_curves, z=50, p = 0.001)
#tokeep = [0, 2, 4, 5]

tcurves 							= tuning_curves[tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]


#################
# OPTO
opto_ep 							= loadOptoEp(data_directory, epoch=0, n_channels = 2, channel = 0)
opto_ep 							= opto_ep.merge_close_intervals(40000)
frates, rasters, bins, stim_duration = computeRasterOpto(spikes, opto_ep, 1000)

opto_ep = sws_ep.intersect(opto_ep)

nopto_ep = nts.IntervalSet(
	start = opto_ep['start'] - (opto_ep['end'] - opto_ep['start']),
	end = opto_ep['start']
	)



###################
# CORRELATION
wak_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, wake_ep, 200, 3))
nopto_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, nopto_ep, 10, 3))
opto_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, opto_ep, 10, 3))

r_wak = np.corrcoef(wak_rate.T)[np.triu_indices(len(tokeep),1)]
r_nopto = np.corrcoef(nopto_rate.T)[np.triu_indices(len(tokeep),1)]
r_opto = np.corrcoef(opto_rate.T)[np.triu_indices(len(tokeep),1)]

r = pd.DataFrame(data = np.vstack((r_wak, r_nopto, r_opto)).T)

pairs = list(combinations(tokeep, 2))

r.index = pd.Index(pairs)
r.columns = pd.Index(['wak', 'nopto', 'opto'])



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue']
shank = shank.flatten()


figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		# plot(tuning_curves2[1][i], '--', color = colors[shank[i]-1])
		if i in tokeep:
			plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1], linewidth = 3)
		# legend()
		count+=1
		gca().set_xticklabels([])


figure()
subplot(121)
plot(r['wak'], r['opto'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(r['wak'].values, r['opto'].values, 1)
x = np.linspace(r['wak'].min(), r['wak'].max(),5)
plot(x, x*m + b, label = 'opto r='+str(np.round(m,3)), color = 'red')
xlabel('wake')
plot(r['wak'], r['nopto'], 'o', color = 'grey', alpha = 0.5)
m, b = np.polyfit(r['wak'].values, r['nopto'].values, 1)
x = np.linspace(r['wak'].min(), r['wak'].max(),5)
plot(x, x*m + b, label = 'non-opto r='+str(np.round(m,3)), color = 'grey')
legend()
subplot(122)

[plot([0,1],r.loc[p,['nopto', 'opto']], 'o-') for p in r.index]
xticks([0,1], ['sws', 'opto'])

legend()


figure()

for n in peaks.index.values:
	plot(spikes[n].restrict(opto_ep).fillna(peaks.loc[n]), '|', color = 'red')
for n in peaks.index.values:
	plot(spikes[n].restrict(nopto_ep).fillna(peaks.loc[n]), '|', color = 'grey')

[axvspan(opto_ep.loc[i,'start'],opto_ep.loc[i,'end'], alpha = 0.1, color = 'red') for i in opto_ep.index]



# from pyglmnet import GLM


# def compute_GLM_CrossCorrs(spks, ep, bin_size=10, lag=5000, lag_size=50, sigma = 15):
# 	bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
# 	order = list(spks.keys())
# 	time_lag = np.arange(-lag, lag, lag_size, dtype = np.int)

# 	shift = time_lag//bin_size

# 	# all spike counts
# 	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = order)
# 	for k in spike_counts:
# 		spike_counts[k] = np.histogram(spks[k].restrict(ep).as_units('ms').index.values, bins)[0]

# 	spike_counts = nts.TsdFrame(t = spike_counts.index.values, d = spike_counts.values, time_units = 'ms')
# 	spike_counts = spike_counts.restrict(ep)
# 	spike_counts = spike_counts.as_dataframe()
# 	glmcc = pd.DataFrame(index = time_lag, columns = list(combinations(order, 2)), dtype = np.float32)

# 	count = 0
# 	for pair in glmcc.columns:
# 		print(count/len(glmcc.columns))
# 		count += 1
# 		# computing predictors	
# 		X1 = spike_counts[pair[1]]	# predictor 
# 		X2 = spike_counts.drop(list(pair), axis = 1).sum(1) # population
# 		X1 = X1.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = sigma)
# 		X2 = X2.rolling(window=100, win_type='gaussian', center= True, min_periods=1).mean(std = 3)
# 		X_train = pd.concat([X1,X2], axis = 1)
# 		X_train = X_train - X_train.mean(0)
# 		X_train = X_train / X_train.std(0)

# 		# target at different time lag				
# 		b = []
# 		for t, s in zip(time_lag, shift):
# 			glm = GLM(distr="poisson", reg_lambda=0.1, solver = 'cdfast', score_metric = 'deviance', max_iter = 200)
# 			# y_train = np.histogram(spks[pair[0]].restrict(ep).as_units('ms').index.values+t, bins)[0]
# 			y_train = spike_counts[pair[0]].values			
# 			if s < 0 : # backward
# 				y_train = y_train[-s:]
# 				Xtrain = X_train.values[:s]
# 			elif s > 0 : # forward
# 				y_train = y_train[:-s]
# 				Xtrain = X_train.values[s:]
# 			else:
# 				Xtrain = X_train.values

# 			# sys.exit()
# 			# fit glm			
# 			glm.fit(Xtrain, y_train)
# 			b.append(np.float32(glm.beta_[0]))


# 		glmcc[pair] = np.array(b)

# 	return glmcc

# cc_wak = compute_GLM_CrossCorrs(spikes, wake_ep, bin_size=100, lag=2000, lag_size=10)
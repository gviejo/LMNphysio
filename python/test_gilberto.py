import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import pickle5 as pickle
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


data_directory_load = '/home/guillaume/Downloads/my_data'


spikes = pickle.load(open(data_directory_load + '/spikes.pickle', 'rb'))
shank = pickle.load(open(data_directory_load  + '/shank.pickle', 'rb'))
episodes = pickle.load(open(data_directory_load + '/episodes.pickle', 'rb'))
position = pickle.load(open(data_directory_load  + '/position.pickle', 'rb'))
wake_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))



cc = compute_GLM_CrossCorrs({0:spikes[51],1:spikes[68]}, wake_ep, bin_size=10, lag=5000, lag_size=50, sigma = 15)
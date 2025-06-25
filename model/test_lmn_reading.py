# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-06-14 21:08:45
# @Last Modified by:   gviejo
# @Last Modified time: 2025-06-16 21:49:42
import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.pyplot import *
import os, sys
import _pickle as cPickle

def sigmoide(X, beta):
	return 1/(1+np.exp(-beta*X))


def get_spikes(rate):
	'''Rate shuld be a tsdframe'''	
	spikes = {}
	for n in rate.columns:	
		count = np.random.poisson(rate.loc[n])	
		to_sample = np.where(count)[0]
		if len(to_sample):
			spk = np.unique(np.hstack([np.sort(np.random.uniform(rate.t[idx], rate.t[idx+1], count[idx])) for idx in to_sample]))
		else:
			spk = []
		spikes[n] = nap.Ts(t=spk, time_support=rate.time_support)

	return nap.TsGroup(spikes)



dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"

data = nap.load_file(dropbox_path+"/A5011-201014A"+".nwb")

spikes = data['units']
sws_ep = data['sws']

spikes = spikes.getby_threshold("rate", 1.0)

spikes = spikes[spikes.location=="lmn"]

bin_size = 0.02
count = spikes.count(bin_size, sws_ep)
rate = count/bin_size

rate = rate.smooth(std=bin_size*1, windowsize=bin_size*20).as_dataframe()

def sigmoide(x, beta):
    return x.mean()/(1+np.exp(-beta*(x-x.mean()/2)))

def sigmoid(x, k=1.0):
    return 1 / (1 + np.exp(-k * x))

def morph_sigmoid_to_linear(x, alpha=0.0):
    """
    alpha=0.0 -> sigmoid
    alpha=1.0 -> linear
    """
    s = sigmoide(x, 1)
    # l = (x - x.min()) / (x.max() - x.min())  # Normalize linear to [0,1]
    l = x
    return (1 - alpha) * s + alpha * l




plot(rate[34])

plot(morph_sigmoid_to_linear(rate[34], 0.2), label="alpha=0.2")

plot(morph_sigmoid_to_linear(rate[34], 0.8), label="alpha=0.8")

legend()
show()



# predicted_rate = sigmoide(np.log(rate), 20)

# pspikes = get_spikes(predicted_rate)

# pspikes.order = spikes.order


# n = 34

# figure()
# subplot(211)
# plot(rate.loc[n], '-')
# plot(spikes[n].restrict(ep).fillna(-1), '|')
# subplot(212)
# plot(sigmoide(np.log(rate.loc[n]), 0.001))

# figure()
# x = np.arange(-1000, 1000, 0.001)
# plot(x, sigmoide(x, 0.001))
# show()







# figure()
# ax = subplot(211)
# pcolormesh(np.random.poisson(sigmoide(np.log(rate), 0.01).values.T))
# subplot(212)
# pcolormesh(rate.values.T)
# show()




# figure()

# ax = subplot(211)

# # pcolormesh(np.random.poisson(predicted.values.T))
# plot(pspikes.restrict(ep).to_tsd("order"), '|')

# subplot(212)

# # pcolormesh(rate.values.T)

# plot(spikes.restrict(ep).to_tsd("order"), '|')

# show()


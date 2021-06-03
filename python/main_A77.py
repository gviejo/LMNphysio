import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import pickle5 as pickle

data_directory_load = '/home/guillaume/Downloads/my_data'

# load data
spikes = pickle.load(open(data_directory_load + '/spikes.pickle', 'rb'))
shank = pickle.load(open(data_directory_load  + '/shank.pickle', 'rb'))
episodes = pickle.load(open(data_directory_load + '/episodes.pickle', 'rb'))
position = pickle.load(open(data_directory_load  + '/position.pickle', 'rb'))
wake_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))


spatial_curves, extent				= computePlaceFields(spikes, position[['x', 'z']], wake_ep, 30)

figure()
for i in spikes:
	subplot(6,7,i+1)
	tmp = scipy.ndimage.gaussian_filter(spatial_curves[i], sigma = 2)
	imshow(tmp, extent = extent, interpolation = 'bilinear', cmap = 'jet')
	colorbar()

sp2 = {}
for k in spatial_curves.keys():
	tmp = scipy.ndimage.gaussian_filter(spatial_curves[k], 2)
	sp2[k] = scipy.signal.correlate2d(tmp, tmp)


figure()
for i in spikes:
	subplot(6,7,i+1)	
	imshow(sp2[i], extent = extent, cmap = 'jet', interpolation = 'bilinear')
	colorbar()



autocorr_wake, frate_wake 			= compute_AutoCorrs(spikes, wake_ep)

figure()
for i in spikes:
	subplot(6,7,i+1)
	plot(autocorr_wake[i])
	# plot(autocorr_sleep[i])
	legend()


tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)

figure()
for i in spikes:
	subplot(6,7,i+1, projection = 'polar')
	plot(tuning_curves[i])
	legend()

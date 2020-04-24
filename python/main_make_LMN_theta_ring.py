import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from matplotlib.colors import hsv_to_rgb
import hsluv
from pycircstat.descriptive import mean as circmean
from sklearn.manifold import SpectralEmbedding, Isomap
from umap import UMAP

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = r'D:\Dropbox (Peyrache Lab)\Peyrache Lab Team Folder\Data\LMN'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)

infoall = []
ccufos = []

# for s in datasets:
# for s in datasets:
for s in ['A5000/A5002/A5002-200304A']:
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	episodes  		= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	events 			= list(np.where(episodes == 'wake')[0].astype('str'))
	events			= list(np.where(episodes == 'wake')[0].astype('str'))
	spikes, shank 	= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position		= loadPosition(path, events, episodes)
	wake_ep 		= loadEpoch(path, 'wake', episodes)
	sleep_ep		= loadEpoch(path, 'sleep')
	sws_ep 			= loadEpoch(path, 'sws')
	rem_ep 			= loadEpoch(path, 'rem')
	theta_wake_ep	= loadEpoch(path, 'wake.evt.theta')


	####################################################################################################################
	# BIN WAKE
	####################################################################################################################
	spikes = {n:spikes[n] for n in np.where(shank==6)[0][0:-2]}

	bin_size = 8
	bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)

	neurons = spikes.keys()
	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
	for i in neurons:
		spks = spikes[i].as_units('ms').index.values
		spike_counts[i], _ = np.histogram(spks, bins)

	# rate = np.sqrt(spike_counts/(bin_size*1e-3))
	rate = spike_counts/(bin_size*1e-3)
	rate2 = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=5)
	tmp = nts.TsdFrame(t = rate2.index.values, d = rate2.values, time_units = 'ms').restrict(theta_wake_ep).values
	tmp = tmp.astype(np.float32)
	

	####################################################################################################################
	# BIN PHASE
	####################################################################################################################
	phase = pd.read_hdf(os.path.join(path, 'Analysis', 'phase_theta_wake.h5'))
	phase = nts.Tsd(phase)
	phase = phase.restrict(wake_ep)

	phase2 = phase.groupby(np.digitize(phase.as_units('ms').index.values, bins)).mean().values
	phase2 = (phase2 + 2*np.pi)%(2*np.pi)
	phase2 = nts.Tsd(t = bins[0:-1] + np.diff(bins)/2., d = phase2, time_units = 'ms').restrict(theta_wake_ep)

	H = phase2.values/(2*np.pi)
	HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
	RGB = hsv_to_rgb(HSV)

	# RGB = np.array([hsluv.hsluv_to_rgb(HSV[i]) for i in range(len(HSV))])
	n = 30000

	# sys.exit()
	
	smap = SpectralEmbedding(n_components = 2, n_neighbors=1000).fit_transform(tmp[0:n])
	scatter(smap[:,0], smap[:,1], c = RGB[0:n])
	show()

	sys.exit()
	ump = UMAP(n_neighbors = 100).fit_transform(tmp)
	scatter(ump[:,0], ump[:,1], c = RGB, marker = '.')
	show()

	sys.exit()

	imap = Isomap(n_neighbors = 100, n_components = 2, n_jobs = -1).fit_transform(tmp[0:n])
	scatter(imap[:,0], imap[:,1], c = RGB[0:n], marker = '.')
	show()

	sys.exit()



	figure()
	scatter(ump[:,0], ump[:,1], s = 1)

	show()


	datatosave = {'ump':ump,
					'wakangle':wakangle}

	# import _pickle as cPickle
	# cPickle.dump(datatosave, open('../figures/figures_poster_2019/fig_4_ring_lmn.pickle', 'wb'))

	# from sklearn.manifold import Isomap
	# imap = Isomap(n_neighbors = 100, n_components = 2).fit_transform(tmp)
	# figure()
	# scatter(imap[:,0], imap[:,1], c= RGB, marker = '.', alpha = 0.5, linewidth = 0, s = 100)






	# figure()
	# for i,n in enumerate(tcurves.columns):
	# 	subplot(3,5,i+1, projection = 'polar')
	# 	plot(tcurves[n])
	 



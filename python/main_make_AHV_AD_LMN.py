import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys, os
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from umap import UMAP
from sklearn.decomposition import PCA
import _pickle as cPickle
from pycircstat.descriptive import mean as circmean

data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

# sessions = ['A1407-190416', 'A1407-190417']
sessions = info.loc['A1407-190403':].index.values
# sessions = info.index.values[1:]

lmn_ahv = []
lmn_hdc = []

lmn_hd_info = []

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
	ahv_curves 						= computeAngularVelocityTuningCurves(spikes, position['ry'], wake_ep, 61, norm=False)
	hd_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)

	frate 							= pd.Series(index = spikes.keys(), data = [len(spikes[k].restrict(wake_ep))/wake_ep.tot_length('s') for k in spikes.keys()])
	# hd_curves	 					= hd_curves/frate

	names 							= pd.Index([s+'_'+str(n) for n in spikes.keys()])
	ahv_curves.columns 				= names
	hd_curves.columns				= names

	lmn_ahv.append(ahv_curves)
	lmn_hdc.append(hd_curves)

	
lmn_ahv = pd.concat(lmn_ahv, 1)
lmn_hdc = pd.concat(lmn_hdc, 1)


# ADDING AD AHV TUNING CURVES
AD_data = cPickle.load(open('../figures/figures_poster_2019/Data_AD.pickle', 'rb'))
adn_ahv = AD_data['ahvcurves']
adn_hdc = AD_data['tcurves']

# SMOOTHING
lmn_hdc = smoothAngularTuningCurves(lmn_hdc, window = 20, deviation = 3.0)
adn_hdc = smoothAngularTuningCurves(adn_hdc, window = 20, deviation = 3.0)

lmn_ahv = lmn_ahv.loc[-2.5:2.5].rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
adn_ahv = adn_ahv.loc[-2.5:2.5].rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)



datatosave = {	'lmn_hdc':lmn_hdc,
				'lmn_ahv':lmn_ahv,
				'adn_hdc':adn_hdc,
				'adn_ahv':adn_ahv
				}

cPickle.dump(datatosave, open('../figures/figures_poster_2019/all_tcurves_AD_LMN.pickle', 'wb'))
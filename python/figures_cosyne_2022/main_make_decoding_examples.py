import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
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
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
shanks = pd.read_csv(os.path.join(data_directory,'ADN_LMN_shanks.txt'), header = None, index_col = 0, names = ['ADN', 'LMN'], dtype = np.str)

infos = getAllInfos(data_directory, datasets)


data_directory = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201014A'
s = 'LMN-ADN/A5011/A5011-201014A'

episodes = ['sleep', 'wake', 'sleep']

events = ['1']


spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
acceleration 						= acceleration[[0,1,2]]
acceleration.columns 				= pd.Index(np.arange(3))
newsleep_ep 						= refineSleepFromAccel(acceleration, sleep_ep)
sws_ep 								= loadEpoch(data_directory, 'sws')
rem_ep 								= loadEpoch(data_directory, 'rem')


wake_ep 							= wake_ep.loc[[0]]

tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
tcurves 							= smoothAngularTuningCurves(tuning_curves, 10, 2)
tokeep, stat 						= findHDCells(tcurves)#, z=10, p = 0.001)

adn = np.intersect1d(tokeep, np.hstack([np.where(shank == i)[0] for i in np.fromstring(shanks.loc[s,'ADN'], dtype=int,sep=' ')]))
lmn = np.intersect1d(tokeep, np.hstack([np.where(shank == i)[0] for i in np.fromstring(shanks.loc[s,'LMN'], dtype=int,sep=' ')]))

tokeep 	= np.hstack((adn, lmn))
spikes 	= {n:spikes[n] for n in tokeep}

tcurves 		= tuning_curves[tokeep]
peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))



occupancy 							= np.histogram(position['ry'], np.linspace(0, 2*np.pi, 121), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]

angle_wak, proba_angle_wak,_		= decodeHD(tcurves, spikes, wake_ep, bin_size = 50, px = occupancy)

angle_sleep, proba_angle_sleep,_	= decodeHD(tcurves, spikes, sleep_ep.loc[[1]], bin_size = 50, px = np.ones_like(occupancy))

angle_rem 							= angle_sleep.restrict(rem_ep)

angle_sleep, proba_angle_sleep, spike_counts	= decodeHD(tcurves[adn], {n:spikes[n] for n in adn}, sleep_ep.loc[[0]], bin_size = 30, px = np.ones_like(occupancy))
#angle_sleep, proba_angle_sleep, spike_counts	= decodeHD(tcurves, spikes, sleep_ep.loc[[0]], bin_size = 30, px = np.ones_like(occupancy))
angle_sws 							= angle_sleep.restrict(sws_ep)
	


############################################################################
# SAVINGt
############################################################################
datatosave = {	'wak':angle_wak,
				'rem':angle_rem,
				'sws':angle_sws,
				'tcurves':tcurves,
				'angle':position['ry'],
				'peaks':peaks,
				'proba_angle_sws':proba_angle_sleep,
				'spike_counts':spike_counts

			}

import _pickle as cPickle

cPickle.dump(datatosave, open('../../figures/figures_poster_2021/fig_cosyne_decoding.pickle', 'wb'))


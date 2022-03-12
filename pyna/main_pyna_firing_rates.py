# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 19:20:07
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-09 15:11:15

import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations



############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets_adn = np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
datasets_lmn = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')


frs = {
	'adn': {e:[] for e in ['wak', 'rem', 'sws']},
	'lmn': {e:[] for e in ['wak', 'rem', 'sws']}
}

for st, datasets in zip(['adn', 'lmn'], [datasets_adn, datasets_lmn]):
	for s in datasets:
		print(s)
		############################################################################################### 
		# LOADING DATA
		###############################################################################################
		path = os.path.join(data_directory, s)
		data = nap.load_session(path, 'neurosuite')
		spikes = data.spikes
		position = data.position
		wake_ep = data.epochs['wake']
		sws_ep = data.read_neuroscope_intervals('sws')
		rem_ep = data.read_neuroscope_intervals('rem')
		
		idx = spikes._metadata[spikes._metadata["location"].str.contains(st)].index.values
		spikes = spikes[idx]
		
		############################################################################################### 
		# COMPUTING TUNING CURVES
		###############################################################################################
		tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
		tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
		
		# CHECKING HALF EPOCHS
		wake2_ep = splitWake(position.time_support.loc[[0]])	
		tokeep2 = []
		stats2 = []
		tcurves2 = []	
		for i in range(2):
			tcurves_half = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
			tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

			tokeep, stat = findHDCells(tcurves_half)
			tokeep2.append(tokeep)
			stats2.append(stat)
			tcurves2.append(tcurves_half)		
		tokeep = np.intersect1d(tokeep2[0], tokeep2[1])
		
		spikes = spikes[tokeep]

		for e, ep in zip(['wak', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):
			tmp = spikes.restrict(ep)._metadata['freq']
			tmp.index = pd.Index([data.basename + '_'+str(n) for n in spikes.keys()])
			frs[st][e].append(tmp)

for st in frs.keys():
	for e in frs[st].keys():
		frs[st][e] = pd.concat(frs[st][e])
	frs[st] = pd.DataFrame.from_dict(frs[st])


cPickle.dump(frs, open(os.path.join('../data/', 'All_FR_ADN_LMN.pickle'), 'wb'))

subplot(121)
scatter(frs['adn']['wak'], frs['adn']['sws'])
subplot(122)
scatter(frs['lmn']['wak'], frs['lmn']['sws'])

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-07 10:52:17
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-07 16:51:56
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

isis = {}
frs = {}

for st in ['adn', 'lmn']:
	############################################################################################### 
	# GENERAL infos
	###############################################################################################
	data_directory = '/mnt/DataGuillaume/'
	datasets = np.genfromtxt(os.path.join(data_directory,'datasets_'+st.upper()+'.list'), delimiter = '\n', dtype = str, comments = '#')


	isis[st] = {e:{} for e in ['wak', 'rem', 'sws']}
	frs[st] = {e:[] for e in ['wak', 'rem', 'sws']}

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
		# groups = spikes._metadata.loc[tokeep].groupby("location").groups
		tcurves 		= tuning_curves[tokeep]
		peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

		try:
			velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
			newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
		except:
			velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
			newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)
		
		############################################################################################### 
		# ISI
		###############################################################################################
		name = data.basename			
				
		for e, ep in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep]):
			isi = {}
			fr = spikes.restrict(ep)._metadata["freq"]
			fr.index = pd.Index([name+'_'+str(n) for n in fr.index])
			frs[st][e].append(fr)
			for n in spikes.keys():
				tmp = []
				for j in ep.index.values:
					spk = spikes[n].restrict(ep.loc[[j]]).index.values
					if len(spk)>2:
						tmp.append(np.diff(spk))
				tmp = np.hstack(tmp)
				
				# tmp = np.diff(spikes[n].restrict(ep).index.values)

				isis[st][e][name+'_'+str(n)] = tmp
			
			
for st in frs.keys():
	for e in frs[st].keys():
		frs[st][e] = pd.concat(frs[st][e])

datatosave = {'isis':isis, 'frs':frs}

cPickle.dump(datatosave, open(os.path.join('../data/', 'All_ISI.pickle'), 'wb'))


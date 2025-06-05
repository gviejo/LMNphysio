# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-07 14:55:34
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-06-05 18:05:09
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
import sys
sys.path.append("..")
from functions import *
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations

dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")

data = cPickle.load(open(os.path.join(dropbox_path, 'All_ISI.pickle'), 'rb'))

isis = data['isis']
frs = data['frs']

########################################################################################
# SHORT TERM ISI
########################################################################################

hisi = {}

for st in isis.keys():
	hisi[st] = {}
	for e in isis[st].keys():
		isi = {}
		for n in isis[st][e].keys():
			bins = np.arange(0, 0.1, 0.001)
			tmp = isis[st][e][n]
			weights = np.ones_like(tmp)/float(len(tmp))
			isi[n]= pd.Series(index=bins[0:-1], data=np.histogram(tmp, bins,weights=weights)[0])
		isi = pd.concat(isi, axis=1)
		isi = isi.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
		hisi[st][e] = isi

figure()
count = 1
for st in ['adn', 'lmn']:
	subplot(1,2,count)	
	for i, e in enumerate(['wak', 'rem', 'sws']):			
		m = hisi[st][e].mean(1).loc[0:0.1]
		s = hisi[st][e].std(1).loc[0:0.1]
		plot(m, label = e)
		fill_between(m.index.values, m-s, m+s, alpha = 0.2)		
		title(st)
		legend()
	count +=1


########################################################################################
# LONG TERM ISI
########################################################################################
logisi = {}

for st in isis.keys():
	logisi[st] = {}
	for e in isis[st].keys():
		isi = {}
		for n in isis[st][e].keys():
			# bins = np.arange(0.001, 100.0, 0.001)
			# bins = np.log(bins)
			#bins = np.logspace(n0.001, 100.0, 1000)
			bins = geomspace(0.001, 100.0, 200)
			tmp = isis[st][e][n]
			weights = np.ones_like(tmp)/float(len(tmp))
			isi[n]= pd.Series(index=bins[0:-1], data=np.histogram(tmp, bins,weights=weights)[0])
			#isi[n]= pd.Series(index=bins[0:-1], data=np.histogram(tmp, bins)[0])
		isi = pd.concat(isi, axis=1)
		isi = isi.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
		logisi[st][e] = isi


figure()
gs = GridSpec(3,4)
for j, st in enumerate(['adn', 'lmn']):	
	for i, e in enumerate(['wak', 'rem', 'sws']):
		gs2 = GridSpecFromSubplotSpec(1,2,gs[j,i], width_ratios=[0.2,0.8])
		tmp = logisi[st][e]
		extents = [tmp.index[0], tmp.index[-1], 0, tmp.shape[1]]
		order = frs[st][e].sort_values().index.values[::-1]

		subplot(gs2[0,0])
		plot(frs[st][e][order].values, np.arange(len(order)))
		yticks([])

		subplot(gs2[0,1])
		tmp = tmp[order]
		tmp = tmp/tmp.max()
		imshow(tmp.T, 
			extent = extents,
			origin = 'lower', 
			aspect = 'auto', 
			cmap = 'jet')
		yticks([])		
		if j == 0: title(e)
		if i == 0: ylabel(st)

	subplot(gs[j,-1])
	for i, e in enumerate(['wak', 'rem', 'sws']):
		tmp = logisi[st][e]
		tmp = tmp/tmp.max()
		m = tmp.mean(1)
		s = tmp.std(1)
		semilogx(m.index.values, m.values, label = e)				
		legend()	
for i, e in enumerate(['wak', 'rem', 'sws']):
	gs2 = GridSpecFromSubplotSpec(1,2,gs[-1,i], width_ratios=[0.2,0.8])
	subplot(gs2[0,1])

	for j, st in enumerate(['adn', 'lmn']):
		tmp = logisi[st][e]
		tmp = tmp/tmp.max()
		m = tmp.mean(1)
		s = tmp.std(1)
		semilogx(m.index.values, m.values, label = st)				
	legend()

show()


datatosave = {'hisi':hisi,'logisi':logisi, 'frs':frs}
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'ALL_LOG_ISI.pickle'), 'wb'))

# ########################################################################################
# # RETURN MAP
# ########################################################################################
# rmap = {}

# for st in isis.keys():
# 	rmap[st] = {}
# 	for e in isis[st].keys():
# 		tmp2 = []
# 		for n in isis[st][e].keys():
# 			tmp = isis[st][e][n]
			
# 			bins = np.arange(0.1, 10, 0.01)
# 			# bins = np.log(bins)
				
# 			data = np.histogram2d(
# 				tmp[0:-1],
# 				tmp[1:],
# 				bins = [bins, bins],
# 				weights = np.ones_like(tmp[0:-1])/float(len(tmp[0:-1])))[0]
# 			tmp2.append(data)
# 		tmp2 = np.array(tmp2)
# 		#isi = isi.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
# 		rmap[st][e] = tmp2

# figure()
# count = 1
# for st in ['adn', 'lmn']:	
# 	for i, e in enumerate(['wak', 'rem', 'sws']):			
# 		subplot(2,3,count)	
# 		imshow(rmap[st][e].mean(0), cmap = 'jet', origin='lower')
# 		title(e)
# 		count +=1
# show()

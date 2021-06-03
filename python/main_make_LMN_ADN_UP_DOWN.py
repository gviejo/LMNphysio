#!/usr/bin/env python
'''

'''
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv


############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_KS25.txt'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

datasets = [s for s in datasets if 'A5002' in s or 'A5011' in s]

for s in datasets:
	print(s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
	name 								= s.split('/')[-1]
	path 								= os.path.join(data_directory, s)
	episodes 							= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
	episodes[episodes != 'sleep'] 		= 'wake'
	events								= list(np.where(episodes != 'sleep')[0].astype('str'))	
	spikes, shank 						= loadSpikeData(path)
	n_channels, fs, shank_to_channel 	= loadXML(path)
	position 							= loadPosition(path, events, episodes)
	wake_ep 							= loadEpoch(path, 'wake', episodes)
	sleep_ep 							= loadEpoch(path, 'sleep')					
	sws_ep								= loadEpoch(path, 'sws')
	rem_ep								= loadEpoch(path, 'rem')

	# KEEPING SPIKES ONLY FROM THE THALAMUS
	if 'A5002' in s:
		spikes = {n:spikes[n] for n in np.where(shank<=2)[0]}

	if 'A5011' in s:
		spikes = {n:spikes[n] for n in np.where(shank<=3)[0]}

	#################################################################################################
	#DETECTION UP/DOWN States
	#################################################################################################
	bin_size = 10000 # us
	rates = []
	for e in sws_ep.index:
		ep = sws_ep.loc[[e]]
		bins = np.arange(ep.iloc[0,0], ep.iloc[0,1], bin_size)
		r = pd.DataFrame(index = (bins[0:-1] + np.diff(bins)/2).astype('int'), columns = np.sort(list(spikes.keys())))
		for n in spikes.keys():
			r[n] = np.histogram(spikes[n].restrict(ep).index.values, bins)[0]
		rates.append(r)
	rates = pd.concat(rates, 0)


	total = rates.sum(1)

	total2 = total.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)

	idx = total2[total2<np.percentile(total2,30)].index.values

	tmp = [[idx[0]]]
	for i in range(1,len(idx)):
		if (idx[i] - idx[i-1]) > bin_size:
			tmp.append([idx[i]])
		elif (idx[i] - idx[i-1]) == bin_size:
			tmp[-1].append(idx[i])

	down_ep = np.array([[e[0],e[-1]] for e in tmp if len(e) > 1])
	down_ep = nts.IntervalSet(start = down_ep[:,0], end = down_ep[:,1])



	down_ep = down_ep.drop_short_intervals(bin_size)
	down_ep = down_ep.reset_index(drop=True)
	down_ep = down_ep.merge_close_intervals(bin_size*2)
	down_ep = down_ep.drop_short_intervals(30000)
	down_ep = down_ep.drop_long_intervals(500000)

	down_ep = down_ep.reset_index(drop=True)


	up_ep 	= nts.IntervalSet(down_ep['end'][0:-1], down_ep['start'][1:])
	up_ep = sws_ep.intersect(up_ep)




	###########################################################################################################
	# Writing for neuroscope

	start = down_ep.as_units('ms')['start'].values
	ends = down_ep.as_units('ms')['end'].values

	datatowrite = np.vstack((start,ends)).T.flatten()

	n = len(down_ep)

	texttowrite = np.vstack(((np.repeat(np.array(['PyDown start 1']), n)), 
							(np.repeat(np.array(['PyDown stop 1']), n))
								)).T.flatten()

	evt_file = path+'/'+name+'.evt.py.dow'
	f = open(evt_file, 'w')
	for t, n in zip(datatowrite, texttowrite):
		f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
	f.close()		


	start = up_ep.as_units('ms')['start'].values
	ends = up_ep.as_units('ms')['end'].values

	datatowrite = np.vstack((start,ends)).T.flatten()

	n = len(up_ep)

	texttowrite = np.vstack(((np.repeat(np.array(['PyUp start 1']), n)), 
							(np.repeat(np.array(['PyUp stop 1']), n))
								)).T.flatten()

	evt_file = path+'/'+name+'.evt.py.upp'
	f = open(evt_file, 'w')
	for t, n in zip(datatowrite, texttowrite):
		f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
	f.close()		

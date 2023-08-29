# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-02-28 13:37:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-04 15:41:08
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import sys, os
import pandas as pd
from functions import *

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
# shanks = pd.read_csv(os.path.join(data_directory,'ADN_LMN_shanks.txt'), header = None, index_col = 0, names = ['ADN', 'LMN'], dtype = str)

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
# shanks = pd.read_csv(os.path.join(data_directory,'LMN_shanks.txt'), header = None, index_col = 0, names = ['LMN'], dtype = str)

infos = getAllInfos(data_directory, datasets)

# sys.exit()


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
	print(episodes)

	data = nap.load_session(path, 'neurosuite')

	spikes = data.spikes
	position = data.position
	wake_ep = data.epochs['wake']

	wake_ep = wake_ep.loc[[0]]

	print(spikes)

	sws_ep = data.read_neuroscope_intervals('sws')
	rem_ep = data.read_neuroscope_intervals('rem')	

	# # COMPUTING TUNING CURVES
	# tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position['ry'].time_support.loc[[0]])
	# tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)

	# ############################################################################################### 
	# # FIGURES
	# ###############################################################################################

	# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

	# shank = spikes._metadata['group']
	# figure()
	# count = 1
	# for j in np.unique(shank):
	# 	neurons = np.where(shank == j)[0]
	# 	for k,i in enumerate(neurons):
	# 		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
	# 		plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
	# 		legend()
	# 		count+=1
	# 		gca().set_xticklabels([])
	# show()

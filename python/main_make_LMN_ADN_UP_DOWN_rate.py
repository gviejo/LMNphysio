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
from matplotlib.gridspec import GridSpec

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
shanks = pd.read_csv(os.path.join(data_directory,'ADN_LMN_shanks.txt'), header = None, index_col = 0, names = ['ADN', 'LMN'], dtype = np.str)

infos = getAllInfos(data_directory, datasets)


allmua = []
alladn = []
alllmn = []

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
	down_ep, up_ep 						= loadUpDown(path)

	# Only taking the first wake ep
	wake_ep = wake_ep.loc[[0]]
	
	############################################################################################### 
	# COMPUTING TUNING CURVES
	###############################################################################################
	tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)	
	tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

	# CHECKING HALF EPOCHS
	wake2_ep = splitWake(wake_ep)
	tokeep2 = []
	stats2 = []
	tcurves2 = []
	for i in range(2):
		# tcurves_half = computeLMNAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]])[0][1]
		tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
		tcurves_half = smoothAngularTuningCurves(tcurves_half, 10, 2)
		tokeep, stat = findHDCells(tcurves_half)
		tokeep2.append(tokeep)
		stats2.append(stat)
		tcurves2.append(tcurves_half)

	tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

	# NEURONS FROM ADN	
	adn = np.intersect1d(tokeep, np.hstack([np.where(shank == i)[0] for i in np.fromstring(shanks.loc[s,'ADN'], dtype=int,sep=' ')]))
	lmn = np.intersect1d(tokeep, np.hstack([np.where(shank == i)[0] for i in np.fromstring(shanks.loc[s,'LMN'], dtype=int,sep=' ')]))

	tokeep 	= np.hstack((adn, lmn))
	spikes 	= {n:spikes[n] for n in tokeep}

	tcurves 		= tuning_curves[tokeep]
	peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))



	# TAKING UP_EP AND DOWN_EP LARGER THAN 100 ms
	up_ep = up_ep.drop_short_intervals(200, time_units = 'ms')	
	up_ep = up_ep.drop_long_intervals(3000, time_units = 'ms')	
	down_ep = down_ep.drop_short_intervals(50, time_units = 'ms')
	down_ep = down_ep.drop_long_intervals(1000, time_units = 'ms')	



	######################################################################################################
	# HD RATES / UP DOWN	
	######################################################################################################

	mua = []
	bins = np.hstack((np.linspace(0,1,200)-1,np.linspace(0,1,200)[1:]))	
	for n in spikes.keys():
		spk = spikes[n].restrict(up_ep).index.values
		spk2 = np.array_split(spk, 10)

		start_to_spk = []
		for i in range(len(spk2)):
			tmp1 = np.vstack(spk2[i]) - up_ep['start'].values
			tmp1 = tmp1.astype(np.float32).T
			tmp1[tmp1<0] = np.nan
			start_to_spk.append(np.nanmin(tmp1, 0))
		start_to_spk = np.hstack(start_to_spk)

		spk_to_end = []
		for i in range(len(spk2)):
			tmp2 = np.vstack(up_ep['end'].values) - spk2[i]
			tmp2 = tmp2.astype(np.float32)
			tmp2[tmp2<0] = np.nan
			spk_to_end.append(np.nanmin(tmp2, 0))
		spk_to_end = np.hstack(spk_to_end)

		d = start_to_spk/(start_to_spk+spk_to_end)
		mua_up = d

		spk = spikes[n].restrict(down_ep).index.values
		if len(spk):
			tmp1 = np.vstack(spk) - down_ep['start'].values
			tmp1 = tmp1.astype(np.float32).T
			tmp1[tmp1<0] = np.nan
			start_to_spk = np.nanmin(tmp1, 0)

			tmp2 = np.vstack(down_ep['end'].values) - spk
			tmp2 = tmp2.astype(np.float32)
			tmp2[tmp2<0] = np.nan
			spk_to_end = np.nanmin(tmp2, 0)

			d = start_to_spk/(start_to_spk+spk_to_end)
			mua_down = d
		else:
			mua_down = np.array([])

		p, _ = np.histogram(np.hstack((mua_down-1,mua_up)), bins)

		mua.append(p)


	mua = pd.DataFrame(
		index = bins[0:-1]+np.diff(bins)/2, 
		data = np.array(mua).T,
		columns = [name+'_'+str(n) for n in spikes.keys()])

	allmua.append(mua)
	alladn.append([name+'_'+str(n) for n in adn])
	alllmn.append([name+'_'+str(n) for n in lmn])
	

allmua = pd.concat(allmua, 1)


datatosave = {'adn':allmua[np.hstack(alladn)],
				'lmn':allmua[np.hstack(alllmn)]}
cPickle.dump(datatosave, open(os.path.join('../data/', 'MUA_ADN_LMN_UP_DOWN.pickle'), 'wb'))



allmua2 = allmua/allmua.sum(0)

figure()
gs = GridSpec(2,len(datasets)+1)
for i in range(len(datasets)):
	subplot(gs[0,i])
	plot(allmua[alladn[i]], color = 'red', alpha = 0.5)
	title(datasets[i].split('/')[-1])
	subplot(gs[1,i])
	plot(allmua[alllmn[i]], color = 'green', alpha = 0.5)
subplot(gs[0,-1])
plot(allmua[np.hstack(alladn)], color = 'red', alpha = 0.5)
subplot(gs[1,-1])
plot(allmua[np.hstack(alllmn)], color = 'green', alpha = 0.5)


for i in range(len(datasets)):
	adnmua = allmua[alladn[i]]
	downmeanadn = adnmua.loc[-1:0].mean()
	upmeanadn = adnmua.loc[0:1].mean()


figure()
gs = GridSpec(2,len(datasets)+1)
for i in range(len(datasets)):
	subplot(gs[0,i])
	plot(allmua2[alladn[i]], color = 'red', alpha = 0.5)
	title(datasets[i].split('/')[-1])
	subplot(gs[1,i])
	plot(allmua2[alllmn[i]], color = 'green', alpha = 0.5)
subplot(gs[0,-1])
plot(allmua2[np.hstack(alladn)], color = 'red', alpha = 0.5)
subplot(gs[1,-1])
plot(allmua2[np.hstack(alllmn)], color = 'green', alpha = 0.5)


figure()
for i in range(len(datasets)):
	subplot(2,2,i+1)
	plot(allmua2[alladn[i]].mean(1), color = 'red', alpha = 0.5)
	title(datasets[i].split('/')[-1])	
	plot(allmua2[alllmn[i]].mean(1), color = 'green', alpha = 0.5)
subplot(2,2,4)
plot(allmua2[np.hstack(alladn)].mean(1), color = 'red', alpha = 0.5)
plot(allmua2[np.hstack(alllmn)].mean(1), color = 'green', alpha = 0.5)



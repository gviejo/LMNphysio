import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)


allcc_wak = []
allcc_rem = []
allcc_sws = []
allpairs = []
alltcurves = []
allfrates = []
allvcurves = []
allscurves = []
allpeaks = []


for s in datasets:
	print(s)
	name = s.split('/')[-1]
	path = os.path.join(data_directory, s)
	############################################################################################### 
	# LOADING DATA
	###############################################################################################
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
	if 'A5011' in s:
		adn = np.where(shank <=3)[0]
		lmn = np.where(shank >3)[0]

	adn = np.intersect1d(adn, tokeep)
	lmn = np.intersect1d(lmn, tokeep)

	spikes = {n:spikes[n] for n in tokeep}
			
	############################################################################################### 
	# CROSS CORRELATION
	###############################################################################################
	cc_wak = compute_CrossCorrs(spikes, wake_ep, norm=True)
	cc_rem = compute_CrossCorrs(spikes, rem_ep, norm=True)	
	cc_sws = compute_CrossCorrs(spikes, sws_ep, 2, 2000, norm=True)


	cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_rem = cc_rem.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
	cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)

	tcurves 							= tuning_curves[tokeep]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
	tcurves 							= tcurves[peaks.index.values]
	neurons 							= [name+'_'+str(n) for n in tcurves.columns.values]
	peaks.index							= pd.Index(neurons)
	tcurves.columns						= pd.Index(neurons)

	new_index = [(name+'_'+str(i),name+'_'+str(j)) for i,j in cc_wak.columns]
	cc_wak.columns = pd.Index(new_index)
	cc_rem.columns = pd.Index(new_index)
	cc_sws.columns = pd.Index(new_index)
	pairs = pd.DataFrame(index = new_index, columns = ['ang diff', 'struct'])
	for i,j in pairs.index:
		if i in neurons and j in neurons:
			a = peaks[i] - peaks[j]
			pairs.loc[(i,j),'ang diff'] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))
			if int(i.split('_')[1]) in adn and int(j.split('_')[1]) in lmn:
				pairs.loc[(i,j),'struct'] = 'adn-lmn'
			elif int(i.split('_')[1]) in adn and int(j.split('_')[1]) in adn:
				pairs.loc[(i,j),'struct'] = 'adn-adn'
			elif int(i.split('_')[1]) in lmn and int(j.split('_')[1]) in lmn:
				pairs.loc[(i,j),'struct'] = 'lmn-lmn'

	
	#######################
	# SAVING
	#######################
	alltcurves.append(tcurves)
	allpairs.append(pairs)
	allcc_wak.append(cc_wak[pairs.index])
	allcc_rem.append(cc_rem[pairs.index])
	allcc_sws.append(cc_sws[pairs.index])
	allpeaks.append(peaks)

 
alltcurves 	= pd.concat(alltcurves, 1)
allpairs 	= pd.concat(allpairs, 0)
allcc_wak 	= pd.concat(allcc_wak, 1)
allcc_rem 	= pd.concat(allcc_rem, 1)
allcc_sws 	= pd.concat(allcc_sws, 1)
allpeaks 	= pd.concat(allpeaks, 0)



sess_groups = pd.DataFrame(pd.Series({k:k.split("_")[0] for k in alltcurves.columns.values})).groupby(0).groups


colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(sess_groups)))

datatosave = {	'tcurves':alltcurves,
				'sess_groups':sess_groups,
				'cc_wak':allcc_wak,
				'cc_rem':allcc_rem,
				'cc_sws':allcc_sws,
				'pairs':allpairs,
				'peaks':allpeaks
				}

cPickle.dump(datatosave, open(os.path.join('../data', 'All_crosscor_ADN_LMN.pickle'), 'wb'))



from matplotlib import gridspec

##########################################################
# TUNING CURVES
figure()
count = 1
for i, g in enumerate(sess_groups.keys()):
	for j, n in enumerate(sess_groups[g]):
		subplot(13,20,count, projection = 'polar')
		plot(alltcurves[n], color = colors[i])
		# title(n)
		xticks([])
		yticks([])
		count += 1

##########################################################
# CROSS CORR
titles = ['wake', 'REM', 'NREM']
figure()
gs = gridspec.GridSpec(3, 5)

for i, st in enumerate(['adn-adn', 'adn-lmn', 'lmn-lmn']):
	group = allpairs[allpairs['struct']==st].sort_values(by='ang diff').index.values
	subplot(gs[i,0])
	plot(allpairs.loc[group, 'ang diff'].values, np.arange(len(group))[::-1])
	ylabel(st)
	for j, cc in enumerate([allcc_wak, allcc_rem, allcc_sws]):
		subplot(gs[i,j+1])
		tmp = cc[group]		
		tmp = tmp - tmp.mean(0)
		tmp = tmp / tmp.std(0)
		tmp = scipy.ndimage.gaussian_filter(tmp.T, (2, 2))

		imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
		
		title(titles[j])
		xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])

	subplot(gs[i,-1])
	cc = allcc_sws[group]
	cc = cc - cc.mean(0)
	cc = cc / cc.std(0)
	cc = cc.loc[-50:50]
	tmp = scipy.ndimage.gaussian_filter(cc.T.values, (1, 1))
	# tmp = cc.loc[-50:50].T
	imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
	
	title(titles[j])
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])




figure()
gs = gridspec.GridSpec(2, 2)


group = allpairs[allpairs['struct']=='adn-lmn'].sort_values(by='ang diff').index.values
subplot(gs[0,0])
cc = allcc_wak[group]
cc = cc - cc.mean(0)
cc = cc / cc.std(0)
cc = cc.loc[-50:50]
tmp = scipy.ndimage.gaussian_filter(cc.T.values, (1, 1))
# tmp = cc.loc[-50:50].T
imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])

subplot(gs[0,1])
plot(cc.mean(1))

subplot(gs[1,0])
for i, gr in enumerate(np.array_split(group, 2)):
	tmp = cc[gr].mean(1)
	tmp = tmp - tmp.mean(0)
	tmp = tmp / tmp.std(0)
	plot(tmp, label = i)
legend()

subplot(gs[1,1])
for i, gr in enumerate(np.array_split(group, 4)):
	tmp = cc[gr].mean(1)
	tmp = tmp - tmp.mean(0)
	tmp = tmp / tmp.std(0)
	plot(tmp, label = i)
legend()

	# subplot(gs[i,-1])
	# cc = allcc_sws[group]
	# cc = cc - cc.mean(0)
	# cc = cc / cc.std(0)
	# cc = cc.loc[-50:50]
	# tmp = scipy.ndimage.gaussian_filter(cc.T.values, (1, 1))
	# # tmp = cc.loc[-50:50].T
	# imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
	
	# title(titles[j])
	# xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])




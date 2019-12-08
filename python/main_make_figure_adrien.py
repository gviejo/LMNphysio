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


# data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# # data_directory 		= '../data/A1400/A1407'
# info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)


# sessions = ['A1407-190416', 'A1407-190417', 'A1407-190422']


# lmn_hdc = []

# lmn_hd_info = []


# for s in sessions:
# 	path = os.path.join(data_directory, s)
# 	############################################################################################### 
# 	# LOADING DATA
# 	###############################################################################################
# 	episodes 							= info.filter(like='Trial').loc[s].dropna().values
# 	events								= list(np.where(episodes == 'wake')[0].astype('str'))
# 	spikes, shank 						= loadSpikeData(path)
# 	n_channels, fs, shank_to_channel 	= loadXML(path)
# 	position 							= loadPosition(path, events, episodes)
# 	wake_ep 							= loadEpoch(path, 'wake', episodes)
# 	sleep_ep 							= loadEpoch(path, 'sleep')					

# 	############################################################################################### 
# 	# COMPUTING TUNING CURVES
# 	###############################################################################################
# 	hd_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)

# 	tokeep, stat 					= findHDCells(hd_curves)	

# 	# frate 							= pd.Series(index = spikes.keys(), data = [len(spikes[k].restrict(wake_ep))/wake_ep.tot_length('s') for k in spikes.keys()])
# 	# hd_curves	 					= hd_curves/frate

# 	names 							= pd.Index([s+'_'+str(n) for n in spikes.keys()])	
# 	hd_curves.columns				= names
# 	hdinfo 							= pd.Series(index=names, data = 0)
# 	hdinfo.loc[names[tokeep]] 		= 1	
# 	lmn_hdc.append(hd_curves)
# 	lmn_hd_info.append(hdinfo)

# lmn_hdc = pd.concat(lmn_hdc, 1)
# lmn_hd_info = pd.concat(lmn_hd_info, 0)

# # ADDING AD TUNING CURVES
# AD_data = cPickle.load(open('../figures/figures_poster_2019/Data_AD_normalized.pickle', 'rb'))
# adn_hdc = AD_data['tcurves']


def get_width(tc):
	tc = smoothAngularTuningCurves(tc, 20, 1)
	centered = {}
	width = {}
	for i in tc.columns:
		# print(i)
		tmp = tc[i]
		idx = tmp.index.values - tmp.idxmax()
		idx[idx<-np.pi] += 2*np.pi
		idx[idx>=np.pi] -= 2*np.pi
		tmp = pd.Series(index = idx, data = tmp.values)
		tmp = tmp.sort_index()
		tmp = tmp - tmp.min()
		tmp = tmp / tmp.max()
		# right = np.abs((tmp[0:]-0.5)).idxmin()
		# left = np.abs((tmp[:0]-0.5)).idxmin()
		tmp = tmp.loc[~tmp.index.duplicated(keep='first')]
		for j in tmp[0:].index.values:
			# print(tmp.loc[j])
			if tmp.loc[j] < 0.5:
				right = j
				break		
		for j in tmp[:0].index.values[::-1]:			
			if tmp.loc[j] < 0.5:
				left = j
				break
		width[i] = right - left
		centered[i] = tmp
	# centered = pd.DataFrame(centered)
	width = pd.Series(width)
	return width

alltcurves = cPickle.load(open('../figures/figures_poster_2019/all_tcurves_AD_LMN.pickle', 'rb'))

adn_hdc = alltcurves['adn_hdc']
lmn_hdc = alltcurves['lmn_hdc']

tokeep, stat 					= findHDCells(lmn_hdc)	
lmn_hdc = lmn_hdc.iloc[:,tokeep]

sub_deep_hdc = scipy.io.loadmat('../figures/figures_poster_2019/TCdeep.mat')['TC']
sub_deep_hdc = pd.DataFrame(index = np.linspace(0, 2*np.pi, 360), data = sub_deep_hdc)
sub_supe_hdc = scipy.io.loadmat('../figures/figures_poster_2019/TCsup.mat')['TC']
sub_supe_hdc = pd.DataFrame(index = np.linspace(0, 2*np.pi, 360), data = sub_supe_hdc)

width = {}
width_mean = {}
width_std = {}
for case, hdc in zip(['lmn', 'adn', 'supe', 'deep'], [lmn_hdc, adn_hdc, sub_supe_hdc, sub_deep_hdc]):	
	width[case] = get_width(hdc.copy())
	width_mean[case] = width[case].mean()
	width_std[case] = width[case].sem()
width_mean = pd.Series(width_mean)
width_std = pd.Series(width_std)




figure()
for i in range(16):
	subplot(4,4,i+1)
	plot(lmn_hdc.iloc[:,i])
	xlim(0, 2*np.pi)

tight_layout()
savefig("../figures/figures_poster_2019/fig_adrien_lmn.pdf", dpi = 900, facecolor = 'white')
# os.system("evince ../figures/figures_poster_2019/fig_adrien_lmn.pdf &")

figure()
for i in range(16):
	subplot(4,4,i+1)
	plot(adn_hdc.iloc[:,i])
	xlim(0, 2*np.pi)

tight_layout()
savefig("../figures/figures_poster_2019/fig_adrien_adn.pdf", dpi = 900, facecolor = 'white')
# os.system("evince ../figures/figures_poster_2019/fig_adrien_lmn.pdf &")


figure()
for i in range(16):
	subplot(4,4,i+1)
	plot(sub_supe_hdc.iloc[:,i])

tight_layout()
savefig("../figures/figures_poster_2019/fig_adrien_supe.pdf", dpi = 900, facecolor = 'white')
# os.system("evince ../figures/figures_poster_2019/fig_adrien_supe.pdf &")

figure()
for i in range(16):
	subplot(4,4,i+1)
	plot(sub_deep_hdc.iloc[:,i])

tight_layout()
savefig("../figures/figures_poster_2019/fig_adrien_deep.pdf", dpi = 900, facecolor = 'white')
# os.system("evince ../figures/figures_poster_2019/fig_adrien_deep.pdf &")

figure()
subplot(111)
bar(np.arange(4), width_mean[['lmn', 'adn', 'supe', 'deep']], yerr = width_std[['lmn', 'adn', 'supe', 'deep']])
xticks(np.arange(4), ['LMN', 'ADN', 'Sup post-sub', 'Deep post-sub'])
yticks(np.arange(0, np.pi, np.pi/4), np.arange(0, np.pi, np.pi/4)*360/(2*np.pi))
tight_layout()
savefig("../figures/figures_poster_2019/fig_adrien_bar.pdf", dpi = 900, facecolor = 'white')


os.system("pdftk ../figures/figures_poster_2019/fig_adrien_lmn.pdf ../figures/figures_poster_2019/fig_adrien_adn.pdf ../figures/figures_poster_2019/fig_adrien_supe.pdf ../figures/figures_poster_2019/fig_adrien_deep.pdf ../figures/figures_poster_2019/fig_adrien_bar.pdf cat output ../figures/figures_poster_2019/fig_adrien_all.pdf")
os.system("evince ../figures/figures_poster_2019/fig_adrien_all.pdf &")
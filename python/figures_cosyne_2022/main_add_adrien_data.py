import numpy as np
import pandas as pd
import os
# from matplotlib.pyplot import plot,show,draw
import sys
sys.path.append('/home/guillaume/ThalamusPhysio/python')
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import neuroseries as nts
import sys
import scipy.ndimage.filters as filters
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from functools import reduce
from multiprocessing import Pool
import h5py as hd
from scipy.stats import zscore
from sklearn.manifold import TSNE, SpectralEmbedding
from skimage import filters
from itertools import combinations
from pycircstat.descriptive import mean as circmean

def zscore_rate(rate):
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	return rate

def binSpikeTrain(spikes, epochs, bin_size, std=0):
	if epochs is None:
		start = np.inf
		end = 0
		for n in spikes:
			start = np.minimum(spikes[n].index.values[0], start)
			end = np.maximum(spikes[n].index.values[-1], end)
		epochs = nts.IntervalSet(start = start, end = end)

	bins = np.arange(epochs['start'].iloc[0], epochs['end'].iloc[-1] + bin_size*1000, bin_size*1000)
	rate = []
	for i,n in enumerate(spikes.keys()):
		count, _ = np.histogram(spikes[n].index.values, bins)
		rate.append(count)
	rate = np.array(rate)	
	rate = nts.TsdFrame(t = bins[0:-1]+bin_size//2, d = rate.T)
	if std:
		rate = rate.as_dataframe()
		rate = rate.rolling(window=std*20,win_type='gaussian',center=True,min_periods=1).mean(std=std)	
		rate = nts.TsdFrame(rate)
	rate = rate.restrict(epochs)
	rate.columns = list(spikes.keys())
	return rate


###############################################################################################################
# LOADING DATA
###############################################################################################################
data_directory 	= '/mnt/DataRAID/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

hd_index = []
allr = []

allcc_wak = []
allcc_rem = []
allcc_sws = []
allpairs = []
alltcurves = []
allfrates = []
allvcurves = []
allscurves = []
allpeaks = []



for m in ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']:
	sessions 		= [n.split("/")[1] for n in datasets if m in n]
	for s in sessions:
		generalinfo 		= scipy.io.loadmat(data_directory+m+"/"+s+'/Analysis/GeneralInfo.mat')		
		shankStructure 		= loadShankStructure(generalinfo)
		spikes,shank		= loadSpikeData(data_directory+m+"/"+s+'/Analysis/SpikeData.mat', shankStructure['thalamus'])						
		wake_ep 		= loadEpoch(data_directory+m+'/'+s, 'wake')
		sleep_ep 		= loadEpoch(data_directory+m+'/'+s, 'sleep')
		sws_ep 			= loadEpoch(data_directory+m+'/'+s, 'sws')
		rem_ep 			= loadEpoch(data_directory+m+'/'+s, 'rem')
		#sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
		#sws_ep 			= sleep_ep.intersect(sws_ep)	
		#em_ep 			= sleep_ep.intersect(rem_ep)		
		# hd
		hd_info 			= scipy.io.loadmat(data_directory+m+'/'+s+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		hd_info_neuron		= np.array([hd_info[n] for n in spikes.keys()])		
		for n in np.where(hd_info[list(spikes.keys())])[0]:
			hd_index.append(s+'_'+str(n))


		if np.sum(hd_info_neuron)>5:

			####################################################################################################################
			# TUNING CURVES
			####################################################################################################################
			spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0]}
			neurons 		= np.sort(list(spikeshd.keys()))

			position 		= pd.read_csv(data_directory+s.split('-')[0]+"/"+ s + '/' + s + ".csv", delimiter = ',', header = None, index_col = [0])
			angle 			= nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')
			tcurves 		= computeAngularTuningCurves(spikeshd, angle, wake_ep, nb_bins = 121, frequency = 1/0.0256)

			neurons 		= tcurves.idxmax().sort_values().index.values


			cc_sws = compute_CrossCorrs(spikeshd, sws_ep, 2, 500, norm=True, reverse=True)
			cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
			
			peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
			tcurves 							= tcurves[peaks.index.values]
			neurons 							= [s+'_'+str(n) for n in tcurves.columns.values]
			peaks.index							= pd.Index(neurons)
			tcurves.columns						= pd.Index(neurons)

			name = s
			new_index = [(name+'_'+str(i),name+'_'+str(j)) for i,j in cc_sws.columns]
			cc_sws.columns = pd.Index(new_index)
			pairs = pd.DataFrame(index = new_index, columns = ['ang diff', 'struct'])
			for i,j in new_index:
					a = peaks[i] - peaks[j]
					# pairs.loc[(i,j),'ang diff'] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))
					pairs.loc[(i,j),'ang diff'] = np.abs(np.arctan2(np.sin(a), np.cos(a)))

			pairs['struct'] = 'adn-adn'
			#######################
			# SAVING
			#######################
			alltcurves.append(tcurves)
			allpairs.append(pairs)
			allcc_sws.append(cc_sws[pairs.index])
			allpeaks.append(peaks)



alltcurves 	= pd.concat(alltcurves, 1)
allpairs 	= pd.concat(allpairs, 0)
allcc_sws 	= pd.concat(allcc_sws, 1)
allpeaks 	= pd.concat(allpeaks, 0)

datatosave = {	'tcurves':alltcurves,
				'cc_sws':allcc_sws,
				'pairs':allpairs,
				'peaks':allpeaks
				}


cPickle.dump(datatosave, open(os.path.join('../../data', 'All_crosscor_ADN_adrien.pickle'), 'wb'))

# figure()
# subplot(121)
# plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
# m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
# x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
# plot(x, x*m + b)
# xlabel('wake')
# ylabel('sws')
# title('r = '+str(np.round(m, 3)))

# figure()
# plot(allr['wak'], allr['sws'], 'o', color = 'red', alpha = 0.5)
# m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
# x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
# plot(x, x*m + b)
# xlabel('wake')
# ylabel('sws')
# title('r = '+str(np.round(m, 3)))


# datatosave = {'allr':allr}
# cPickle.dump(datatosave, open(os.path.join('/home/guillaume/LMNphysio/data/', 'All_correlation_ADN.pickle'), 'wb'))

# show()
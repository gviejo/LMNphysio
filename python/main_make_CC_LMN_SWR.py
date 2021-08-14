import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from matplotlib.gridspec import GridSpecFromSubplotSpec

data_directory = '/mnt/Data2/LMN-PSB-2/A3013/A3013-210806A'
episodes = ['sleep', 'wake', 'wake', 'sleep', 'wake', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']
# episodes = ['sleep', 'wake', 'sleep']

events = ['1', '2', '4', '5']




spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory)

sws_ep								= loadEpoch(data_directory, 'sws')

tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves = smoothAngularTuningCurves(tuning_curves, 5, 1)



tokeep, stat = findHDCells(tuning_curves, z = 10, p = 0.001)

rip_ep, rip_tsd 					= loadRipples(data_directory)

cc_rip = compute_EventCrossCorr(spikes, rip_tsd, sws_ep, binsize = 10, nbins = 400, norm=True)
cc_rip2 = cc_rip.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)
###########################################################
# FIGURES
###########################################################

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = shank.flatten()

figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		ax = subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		gs = GridSpecFromSubplotSpec(1,2,ax)
		subplot(gs[0,0], projection = 'polar')
		plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])		
		if i in tokeep:
			plot(tuning_curves[i], color = colors[shank[i]-1], linewidth = 3)
		legend()
		count+=1
		gca().set_xticklabels([])
		subplot(gs[0,1])
		plot(cc_rip2[i].loc[-500:500], color = colors[shank[i]-1])
		axvline(0, alpha = 0.4)


figure()
plot(cc_rip2[tokeep].loc[-500:500])

show()







#####################
# TO COMPARE WITh adn papier






# loading adn cc ripples
def loadSWRMod(path, datasets, return_index = False):
	import _pickle as cPickle
	tmp = cPickle.load(open(path, 'rb'))
	z = []
	index = []
	for session in datasets:
		neurons = np.array(list(tmp[session].keys()))
		sorte = np.array([int(n.split("_")[1]) for n in neurons])
		ind = np.argsort(sorte)			
		for n in neurons[ind]:
			z.append(tmp[session][n])						
		index += list(neurons[ind])
	z = np.vstack(z)
	index = np.array(index)
	if return_index:
		return (z, index)
	else:
		return z

def gaussFilt(X, wdim = (1,)):
	'''
		Gaussian Filtering in 1 or 2d.		
		Made to fit matlab
	'''
	from scipy.signal import gaussian

	if len(wdim) == 1:
		from scipy.ndimage.filters import convolve1d
		l1 = len(X)
		N1 = wdim[0]*10
		S1 = (N1-1)/float(2*5)
		gw = gaussian(N1, S1)
		gw = gw/gw.sum()
		#convolution
		if len(X.shape) == 2:
			filtered_X = convolve1d(X, gw, axis = 1)
		elif len(X.shape) == 1:
			filtered_X = convolve1d(X, gw)
		return filtered_X	
	elif len(wdim) == 2:
		from scipy.signal import convolve2d
		def conv2(x, y, mode='same'):
			return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)			

		l1, l2 = X.shape
		N1, N2 = wdim		
		# create bordered matrix
		Xf = np.flipud(X)
		bordered_X = np.vstack([
				np.hstack([
					np.fliplr(Xf),Xf,np.fliplr(Xf)
				]),
				np.hstack([
					np.fliplr(X),X,np.fliplr(X)
				]),
				np.hstack([
					np.fliplr(Xf),Xf,np.fliplr(Xf)
				]),
			])
		# gaussian windows
		N1 = N1*10
		N2 = N2*10
		S1 = (N1-1)/float(2*5)
		S2 = (N2-1)/float(2*5)
		gw = np.vstack(gaussian(N1,S1))*gaussian(N2,S2)
		gw = gw/gw.sum()
		# convolution
		filtered_X = conv2(bordered_X, gw, mode ='same')
		return filtered_X[l1:l1+l1,l2:l2+l2]
	else :
		print("Error, dimensions larger than 2")
		return

def loadadndatapapier():	
	mappings = pd.read_csv("/mnt/DataRAID/MergedData/MAPPING_NUCLEUS.csv", index_col = 0)
	datasets = np.loadtxt('/mnt/DataRAID/MergedData/'+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
	swr_mod, swr_ses = loadSWRMod('/mnt/DataRAID/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
	nbins 		= 400
	binsize		= 5
	times 		= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	swr_mod 	= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
	swr_mod = swr_mod.drop(swr_mod.columns[swr_mod.isnull().any()].values, axis = 1)
	swr_mod = swr_mod.loc[-500:500]
	neurons = np.intersect1d(swr_mod.columns.values, mappings.index.values)
	hd_neurons = mappings.loc[neurons][mappings.loc[neurons, 'hd'] == 1].index.values
	
	return swr_mod[hd_neurons]

swr_mod_adn = loadadndatapapier()


tokeep = np.where(shank == 9)[0]
#cc_rip2 = cc_rip[tokeep].loc[-500:500]



tmp = cc_rip2[tokeep].values
window_size 	= 2*150//5
window 			= np.ones(int(window_size))*(1/window_size)
tmp2 = []
for i in range(tmp.shape[1]):
	Hm = np.convolve(tmp[:,i], window, 'same')	
	Hstd = np.sqrt(np.var(Hm))	
	tmp2.append((tmp[:,i] - Hm)/Hstd)
tmp2 = np.array(tmp2)	
cc_lmn = pd.DataFrame(index = cc_rip2.index, columns = tokeep, data = tmp2.T)

cc_lmn = cc_lmn.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=1)

cc_lmn = cc_lmn.loc[-500:500]


figure()
subplot(121)
plot(swr_mod_adn.mean(1), color= 'red', label = 'adn papier')
plot(cc_lmn, color = 'green', alpha = 0.5)
plot(cc_lmn.mean(1), color = 'green', alpha = 1, label = 'lmn')
legend()
subplot(122)
plot(swr_mod_adn.mean(1), color = 'red', label = 'adn papier')
plot(cc_lmn.mean(1), color = 'green', label = 'lmn')
legend()


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
data_directory = r'D:\Dropbox (Peyrache Lab)\Peyrache Lab Team Folder\Data\LMN'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.atleast_1d(np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'))
infos = getAllInfos(data_directory, datasets)


# sys.exit()
s = datasets[5] # good lmn
# s = datasets[0]

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

acceleration						= loadAuxiliary(path)
if 'A5001' in s and 3 in acceleration.columns:
	acceleration = acceleration[[3,4,5]]
	acceleration.columns = range(3)
elif 'A5002' in s:
	acceleration = acceleration[[0,1,2]]
newsleep_ep 						= refineSleepFromAccel(acceleration, sleep_ep)


# Only taking the first wake ep
wake_ep = wake_ep.loc[[0]]

# Taking only neurons from LMN
if 'A5002' in s:
	spikes = {n:spikes[n] for n in np.intersect1d(np.where(shank.flatten()>2)[0], np.where(shank.flatten()<4)[0])}		

if 'A5001' in s:
	spikes = {n:spikes[n] for n in np.where(shank.flatten()==6)[0]}		


############################################################################################### 
# COMPUTING TUNING CURVES
###############################################################################################
# tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
tuning_curves = {1:computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)}
for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)

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
tokeep2 = np.union1d(tokeep2[0], tokeep2[1])

# tokeep = np.arange(67, 83)

figure()
for i, n in enumerate(spikes.keys()):
	subplot(5,5,i+1, projection = 'polar')
	plot(tuning_curves[1][n], label = n)
	if n in tokeep2:
		plot(tcurves2[0][n])
		plot(tcurves2[1][n])
	if n in tokeep:
		plot(tuning_curves[1][n], color = 'red')
	legend()



spikes = {n:spikes[n] for n in tokeep}
		
############################################################################################### 
# CROSS CORRELATION
###############################################################################################
cc_wak = compute_CrossCorrs(spikes, wake_ep, norm=True)
cc_rem = compute_CrossCorrs(spikes, rem_ep, norm=True)
cc_sws = compute_CrossCorrs(spikes, sws_ep, 0.5, 800, norm=True)

cc_wak = cc_wak.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
cc_rem = cc_rem.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
cc_sws = cc_sws.rolling(window=10, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)



tcurves 							= tuning_curves[1][tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]
neurons 							= [name+'_'+str(n) for n in tcurves.columns.values]
peaks.index							= pd.Index(neurons)
tcurves.columns						= pd.Index(neurons)



new_index = [(name+'_'+str(i),name+'_'+str(j)) for i,j in cc_wak.columns]
cc_wak.columns = pd.Index(new_index)
cc_rem.columns = pd.Index(new_index)
cc_sws.columns = pd.Index(new_index)
pairs = pd.Series(index = new_index, data = np.nan)
for i,j in pairs.index:	
	if i in neurons and j in neurons:
		a = peaks[i] - peaks[j]
		pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))


pairs = pairs.dropna().sort_values()


############################################################################################### 
# DOING BAYESIAN DECODING TO CHECK REM 
###############################################################################################
spikes = {n:spikes[int(n.split('_')[1])] for n in neurons}
occupancy 							= np.histogram(position['ry'], np.linspace(0, 2*np.pi, 121), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]

angle_wak, proba_angle_wak,_		= decodeHD(tcurves, spikes, wake_ep, bin_size = 50, px = occupancy)
# entropy_wak 						= -np.sum(proba_angle_wak*np.log(proba_angle_wak), 1)
# entropy_wak 						= nts.Tsd(t = entropy_wak.index.values, d = entropy_wak.values, time_units = 'ms')

angle_sleep, proba_angle_sleep,_	= decodeHD(tcurves, spikes, sleep_ep.loc[[0]], bin_size = 50, px = np.ones_like(occupancy))
# entropy_sleep 						= -np.sum(proba_angle_sleep*np.log(proba_angle_sleep), 1)
# entropy_sleep 						= nts.Tsd(t = entropy_sleep.index.values, d = entropy_sleep.values, time_units = 'ms')

# angle_rem 							= angle_sleep.restrict(rem_ep)
# entropy_rem 						= -np.sum(proba_angle_wak*np.log(proba_angle_sleep), 1)
# entropy_rem 						= 

# angle_sleep, proba_angle_sleep, spike_counts	= decodeHD(tcurves, spikes, sleep_ep.loc[[1]], bin_size = 10, px = np.ones_like(occupancy))

# angle_sws 							= angle_sleep.restrict(sws_ep)


# ############################################################################################### 
# # LOADING LFP
# ###############################################################################################
if 'A5001' in s:
	lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 84, 1250, 'int16')
elif 'A5002' in s:
	lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 80, 1250, 'int16')
elif 'A1407' in s:
	lfp 		= loadLFP(os.path.join(data_directory,s,name+'.eeg'), n_channels, 1, 1250, 'int16')
lfp 		= downsample(lfp, 1, 5)


# ##################################################################################################
# # DETECTION THETA
# ##################################################################################################
lfp_filt_theta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 4, 12, 1250/5, 2))
power_theta		= nts.Tsd(lfp_filt_theta.index.values, np.abs(lfp_filt_theta.values))
power_theta		= power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=40)

lfp_filt_delta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 0.5, 4, 1250/5, 2))
power_delta		= nts.Tsd(lfp_filt_delta.index.values, np.abs(lfp_filt_delta.values))
power_delta		= power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=40)

ratio 			= nts.Tsd(t = power_theta.index.values, d = np.log(power_theta.values/power_delta.values))

ratio2			= ratio.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=400)
ratio2 			= nts.Tsd(t = ratio2.index.values, d = ratio2.values)



##########################################################
# CROSS CORR
##########################################################
titles = ['wake', 'REM', 'NREM']
figure()
subplot(2,4,1)
plot(pairs.values, np.arange(len(pairs))[::-1])
for i, cc in enumerate([cc_wak, cc_rem, cc_sws]):
	subplot(2,4,i+2)
	tmp = cc[pairs.index].T.values
	imshow(scipy.ndimage.gaussian_filter(tmp, 2), aspect = 'auto', cmap = 'jet')
	title(titles[i])
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])
	subplot(2,3,i+3+1)
	plot(cc)





##########################################################
# DECODING
##########################################################
figure()
ax = subplot(311)
for i,n in enumerate(neurons):
	plot(spikes[n].fillna(peaks[n]), '|', ms = 15, mew = 1)
	# plot(spikes[n].fillna(i), '|', markersize = 2, markeredgewidth=2, linewidth = 100)
plot(position['ry'].restrict(wake_ep))
# plot(angle_sleep.restrict(newsleep_ep))
ylim(0, 2*np.pi)

subplot(312, sharex = ax)
[plot(lfp.restrict(rem_ep.loc[[i]]), color = 'blue') for i in rem_ep.index]
[plot(lfp.restrict(sws_ep.loc[[i]]), color = 'orange') for i in sws_ep.index]
plot(lfp_filt_theta.restrict(newsleep_ep))
# plot(lfp_filt_delta.restrict(newsleep_ep))



subplot(313, sharex = ax)
[plot(ratio.restrict(rem_ep.loc[[i]]), color = 'blue') for i in rem_ep.index]
[plot(ratio.restrict(sws_ep.loc[[i]]), color = 'orange') for i in sws_ep.index]
plot(ratio2.restrict(newsleep_ep))
ylabel('log(theta)/log(delta)')


show()


##########################################################
# TEST UMAP REM
##########################################################
bin_size = 50
bins = np.arange(rem_ep.as_units('ms').start.iloc[0], rem_ep.as_units('ms').end.iloc[-1], bin_size)

spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = tcurves.columns)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

rate_wake = np.sqrt(spike_counts/(bin_size*1e-3))

tmp = rate_wake.rolling(window=20,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
tmp.index = tmp.index * 1000
tmp.index = np.int64(tmp.index)
tmp = nts.TsdFrame(tmp)

tmp = tmp.restrict(rem_ep)

# from umap import UMAP
# ump = UMAP(n_neighbors = 100).fit_transform(tmp.values)

# figure()
# scatter(ump[:,0].flatten(), ump[:,1].flatten())
# show()


# from sklearn.manifold import Isomap
# imap = Isomap(n_neighbors = 100).fit_transform(tmp.values)

# figure()
# scatter(imap[:,0].flatten(), imap[:,1].flatten())
# show()



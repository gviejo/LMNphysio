import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201011A'

episodes = ['sleep', 'wake', 'sleep']

events = ['1']


spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
acceleration 						= acceleration[[0,1,2]]
acceleration.columns 				= pd.Index(np.arange(3))
newsleep_ep 						= refineSleepFromAccel(acceleration, sleep_ep)


sleepscoring = scipy.io.loadmat(data_directory + '/A5011-201011A.SleepState.states.mat')
timeindex = sleepscoring['SleepState'][0][0][1][0][0][1].flatten()

sws_ep = loadEpoch(data_directory, 'sws')


sys.exit()


##################################################################################################
# LOADING ONE CHANNEL ONLY AND DOWNSAMPLING IT
##################################################################################################
lfp 		= loadLFP(data_directory+'/'+data_directory.split('/')[-1]+'.dat', n_channels, 66, 20000, 'int16')
lfp 		= downsample(lfp, 1, 80)


# lfp 		= loadLFP(data_directory+'/'+data_directory.split('/')[-1]+'.dat', n_channels, 65, 1250, 'int16')
# lfp 		= downsample(lfp, 1, 5)

sys.exit()


##################################################################################################
# DETECTION THETA
##################################################################################################
lfp_filt_theta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 4, 12, 1250/5, 2))
power_theta		= nts.Tsd(lfp_filt_theta.index.values, np.abs(lfp_filt_theta.values))
# power_theta		= power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=40)
power_theta		= power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

lfp_filt_delta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 0.5, 4, 1250/5, 2))
power_delta		= nts.Tsd(lfp_filt_delta.index.values, np.abs(lfp_filt_delta.values))
# power_delta		= power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=40)
power_delta		= power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

ratio 			= nts.Tsd(t = power_theta.index.values, d = np.log(power_theta.values/power_delta.values))

ratio2			= ratio.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=200)
ratio2 			= nts.Tsd(t = ratio2.index.values, d = ratio2.values)


index 			= (ratio2.as_series() > 0).values*1.0
start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
tmp 			= np.where(end_cand != start_cand)
start_cand 		= ratio2.index.values[start_cand[tmp]]
end_cand	 	= ratio2.index.values[end_cand[tmp]]
good_ep			= nts.IntervalSet(start_cand, end_cand)
good_ep			= newsleep_ep.intersect(good_ep)
# good_ep			= good_ep.drop_short_intervals(5, time_units = 's')
good_ep			= good_ep.merge_close_intervals(10, time_units = 's')
good_ep			= good_ep.drop_short_intervals(5, time_units = 's')
good_ep			= good_ep.reset_index(drop=True)
# good_ep			= good_ep.merge_close_intervals(0.5, time_units = 's')



theta_rem_ep	= good_ep

sws_ep 	= newsleep_ep.set_diff(theta_rem_ep)
sws_ep = sws_ep.merge_close_intervals(0).drop_short_intervals(0)

figure()
ax = subplot(211)
[plot(lfp.restrict(theta_rem_ep.loc[[i]]), color = 'blue') for i in theta_rem_ep.index]
[plot(lfp.restrict(sws_ep.loc[[i]]), color = 'orange') for i in sws_ep.index]
plot(lfp_filt_theta.restrict(newsleep_ep))
subplot(212, sharex = ax)
[plot(ratio.restrict(theta_rem_ep.loc[[i]]), color = 'blue') for i in theta_rem_ep.index]
[plot(ratio.restrict(sws_ep.loc[[i]]), color = 'orange') for i in sws_ep.index]
plot(ratio2.restrict(newsleep_ep))

axhline(0)
show()

# sys.exit()

writeNeuroscopeEvents(data_directory+'/'+data_directory.split('/')[-1]+'.rem.evt', theta_rem_ep, "Theta")
writeNeuroscopeEvents(data_directory+'/'+data_directory.split('/')[-1]+'.sws.evt', sws_ep, "SWS")



sys.exit()

phase 			= getPhase(lfp_hpc, 6, 14, 16, fs/5.)	
ep 				= { 'wake'	: theta_wake_ep,
					'rem'	: theta_rem_ep}
theta_mod 		= {}



for e in ep.keys():		
	spikes_phase	= {n:phase.realign(spikes[n], align = 'closest') for n in spikes.keys()}

	# theta_mod[e] 	= np.ones((n_neuron,3))*np.nan
	theta_mod[e] 	= {}
	for n in range(len(spikes_phase.keys())):			
		neuron = list(spikes_phase.keys())[n]
		ph = spikes_phase[neuron].restrict(ep[e])
		mu, kappa, pval = getCircularMean(ph.values)
		theta_mod[e][session.split("/")[1]+"_"+str(neuron)] = np.array([mu, pval, kappa])
		spikes_theta_phase[e][session.split("/")[1]+"_"+str(neuron)] = ph.values


stop = time.time()
print(stop - start, ' s')		
datatosave[session] = theta_mod


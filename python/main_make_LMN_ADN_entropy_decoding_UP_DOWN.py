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

data_directory = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201014A'


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
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)
sws_ep 								= loadEpoch(data_directory, 'sws')
rem_ep 								= loadEpoch(data_directory, 'rem')
down_ep, up_ep 						= loadUpDown(data_directory)

############################################################################################### 
# COMPUTING TUNING CURVES
###############################################################################################
tuning_curves, velocity, bins_velocity = computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 31)
for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 10, 2)


tokeep, stat = findHDCells(tuning_curves[1], z = 20, p = 0.001 , m = 1)

adn = list(np.where(shank <= 3)[0])
lmn = list(np.where(shank == 5)[0])
adn = np.intersect1d(adn, tokeep)
lmn = np.intersect1d(lmn, tokeep)

if 'A5011-201014A' in data_directory:
	adn = adn[:-3]



tcurvesall = {}
peaksall = {}
for neurons, name in zip([adn, lmn], ['adn', 'lmn']):
	tcurves 							= tuning_curves[1][neurons]
	peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()
	tcurves 							= tcurves[peaks.index.values]
	tcurvesall[name] 					= tcurves
	peaksall[name]						= peaks


############################################################################
# DECODING
############################################################################
bin_size 							= 10
occupancy 							= np.histogram(position['ry'], np.linspace(0, 2*np.pi, 31), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
rawsleep_ep 						= loadEpoch(data_directory, 'sleep')
proba_adn 							= []
proba_lmn							= []
for i in range(2):
	tmp1, tmp2, _ = decodeHD(tcurvesall['adn'], spikes, rawsleep_ep.loc[[i]], bin_size = bin_size, px = np.ones_like(occupancy))
	proba_adn.append(tmp2)
	tmp1, tmp2, _ = decodeHD(tcurvesall['lmn'], spikes, rawsleep_ep.loc[[i]], bin_size = bin_size, px = np.ones_like(occupancy))
	proba_lmn.append(tmp2)


proba_adn = pd.concat(proba_adn, 0)
proba_lmn = pd.concat(proba_lmn, 0)

entropy_adn 						= -np.sum(proba_adn*np.log(proba_adn), 1)
entropy_adn 						= nts.Tsd(t = entropy_adn.index.values, d = entropy_adn.values, time_units = 'ms')

entropy_lmn 						= -np.sum(proba_lmn*np.log(proba_lmn), 1)
entropy_lmn 						= nts.Tsd(t = entropy_lmn.index.values, d = entropy_lmn.values, time_units = 'ms')

post_adn 							= nts.Tsd(t = proba_adn.index.values, d = proba_adn.max(1).values, time_units = 'ms')
post_lmn 							= nts.Tsd(t = proba_lmn.index.values, d = proba_lmn.max(1).values, time_units = 'ms')


###########################################################################
# CROSS CORR UP/ENTROPY
###########################################################################
cce_adn = []
cce_lmn = []
for e in up_ep['start']:
	t = entropy_adn.index[np.argmin(np.abs(entropy_adn.index.values - e))]
	cce_adn.append(entropy_adn.loc[t-200000:t+200000].values)
	cce_lmn.append(entropy_lmn.loc[t-200000:t+200000].values)

cce_adn = np.array([i for i in cce_adn if len(i) == int(200/bin_size)*2+1])
cce_lmn = np.array([i for i in cce_lmn if len(i) == int(200/bin_size)*2+1])

###########################################################################
# CROSS CORR UP/POSTERIOR
###########################################################################
ccp_adn = []
ccp_lmn = []
for e in up_ep['start']:
	t = post_adn.index[np.argmin(np.abs(post_adn.index.values - e))]
	ccp_adn.append(post_adn.loc[t-200000:t+200000].values)
	ccp_lmn.append(post_lmn.loc[t-200000:t+200000].values)

ccp_adn = np.array([i for i in ccp_adn if len(i) == int(200/bin_size)*2+1])
ccp_lmn = np.array([i for i in ccp_lmn if len(i) == int(200/bin_size)*2+1])

######################################################################################################
# HD RATES / UP DOWN	
######################################################################################################

mua = {'adn':[], 'lmn':[]}
bins = np.hstack((np.linspace(0,1,200)-1,np.linspace(0,1,200)[1:]))
for g, index in zip(['adn', 'lmn'], [adn, lmn]):
	for n in index:
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

		p, _ = np.histogram(np.hstack((mua_down-1,mua_up)), bins)

		mua[g].append(p)

	mua[g] = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = np.array(mua[g]).sum(0))
	mua[g] = mua[g]/mua[g].sum()

######################################################################################################
# non HD RATES / UP DOWN	
######################################################################################################
thn = np.setdiff1d(np.where(shank<=3)[0], adn)
mbn = np.setdiff1d(np.where(shank>3)[0], lmn)
mua_nohd = {'th':[], 'mb':[]}
bins = np.hstack((np.linspace(0,1,200)-1,np.linspace(0,1,200)[1:]))
for g, index in zip(['th', 'mb'], [thn, mbn]):
	for n in index:
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

		p, _ = np.histogram(np.hstack((mua_down-1,mua_up)), bins)

		mua_nohd[g].append(p)

	mua_nohd[g] = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = np.array(mua_nohd[g]).sum(0))
	mua_nohd[g] = mua_nohd[g]/mua_nohd[g].sum()



############################################################################
# PLOT
############################################################################
import matplotlib.gridspec as gridspec


adn = peaksall['adn'].index.values
lmn = peaksall['lmn'].index.values

fig = figure()
outergs = GridSpec(3,1, figure = fig, wspace = 0.1)
# SPIKES
ax = subplot(outergs[0,0])
for i,n in enumerate(adn):
	clr = hsluv.hsluv_to_rgb([peaksall['adn'][n]*180/np.pi,85,45])
	plot(spikes[n].restrict(sws_ep).fillna(peaksall['adn'][n]), '|', markersize = 10, color = clr)
ax2 = ax.twinx()
[ax.axvline(up_ep.loc[i,'start'], color = 'blue', alpha = 0.6) for i in up_ep.index]
[ax.axvline(up_ep.loc[i,'end'], color = 'cyan', alpha = 0.6) for i in up_ep.index]

ax3 = subplot(outergs[1,0], sharex = ax)	
for i,n in enumerate(lmn):
	clr = hsluv.hsluv_to_rgb([peaksall['lmn'][n]*180/np.pi,85,45])
	plot(spikes[n].restrict(sws_ep).fillna(peaksall['lmn'][n]), '|', markersize = 10, color = clr)
ax4 = ax3.twinx()
[ax3.axvline(up_ep.loc[i,'start'], color = 'blue', alpha = 0.6) for i in up_ep.index]
[ax3.axvline(up_ep.loc[i,'end'], color = 'cyan', alpha = 0.6) for i in up_ep.index]

ax5 = subplot(outergs[2,0], sharex = ax)
plot(entropy_lmn, label = 'lmn', color = 'green')
ax6 = ax5.twinx()
plot(entropy_adn, color = 'red', label = 'adn')
fig.legend()





times = np.arange(-200, 200+bin_size, bin_size)
mcce_adn = pd.Series(index = times, data = cce_adn.mean(0))
mcce_lmn = pd.Series(index = times, data = cce_lmn.mean(0))

dmcce_adn = pd.Series(index = times[0:-1]+5, data = np.diff(mcce_adn.values))
dmcce_lmn = pd.Series(index = times[0:-1]+5, data = np.diff(mcce_lmn.values))

times = np.arange(-200, 200+bin_size, bin_size)
mccp_adn = pd.Series(index = times, data = ccp_adn.mean(0))
mccp_lmn = pd.Series(index = times, data = ccp_lmn.mean(0))





fig = figure()
ax = fig.add_subplot(221)
plot(mua['adn'], color = 'red', label = 'ADN')
plot(mua_nohd['th'], '--', color= 'red', alpha = 0.6, label = 'TH non-HD')
plot(mua['lmn'], color = 'green', label = 'LMN')
plot(mua_nohd['mb'], '--', color= 'green', alpha = 0.6, label = 'MB non-HD')
ylabel("MUA")
ax.legend(bbox_to_anchor=(0.9, 0.9))
ax = fig.add_subplot(222)
ax.plot(mcce_lmn.loc[-100:100], label = 'lmn', color = 'green')
ax2 = ax.twinx()
ax2.plot(mcce_adn.loc[-100:100], label = 'adn', color = 'red')
axvline(0)
ax.set_xlabel("Time from Up onset")
ax.set_ylabel("Entropy")
ax = fig.add_subplot(223)
ax.plot(mccp_lmn.loc[-100:100], label = 'lmn', color = 'green')
ax2 = ax.twinx()
ax2.plot(mccp_adn.loc[-100:100], label = 'adn', color = 'red')
axvline(0)
ax.set_xlabel("Time from Up onset")
ax.set_ylabel("Posterior")



spikes_adn = np.hstack([spikes[n].index.values for n in adn])
spikes_adn = nts.Ts(t = np.sort(spikes_adn))

spikes_lmn = np.hstack([spikes[n].index.values for n in lmn])
spikes_lmn = nts.Ts(t = np.sort(spikes_lmn))

cc_mua_wak = compute_PairCrossCorr({0:spikes_lmn, 1:spikes_adn}, wake_ep, (0,1), binsize=5, nbins = 1000)
cc_mua_rem = compute_PairCrossCorr({0:spikes_lmn, 1:spikes_adn}, rem_ep, (0,1), binsize=5, nbins = 1000)
cc_mua_sws = compute_PairCrossCorr({0:spikes_lmn, 1:spikes_adn}, sws_ep, (0,1), binsize=5, nbins = 400)
figure()
subplot(221)
plot(cc_mua_wak)
title('wake')
ylabel("LMN/ADN")
subplot(222)
plot(cc_mua_rem)
title('rem')
subplot(223)
plot(cc_mua_sws)
title('sws')

# sys.exit()11

# bins = np.linspace(0, 1, 100)

# entropy_adn_up = []
# entropy_lmn_up = []

# for i in up_ep.index:
# 	tmp = entropy_adn.restrict(up_ep.loc[[i]])
# 	tmp = pd.Series(index = np.linspace(0, 1, len(tmp)), data = tmp.values)
# 	entropy_adn_up.append(np.vstack((np.digitize(tmp.index.values, bins)-1, tmp.values)).T)

# 	tmp = entropy_lmn.restrict(up_ep.loc[[i]])
# 	tmp = pd.Series(index = np.linspace(0, 1, len(tmp)), data = tmp.values)
# 	entropy_lmn_up.append(np.vstack((np.digitize(tmp.index.values, bins)-1, tmp.values)).T)

# entropy_adn_up = np.vstack(entropy_adn_up)
# entropy_lmn_up = np.vstack(entropy_lmn_up)

# tmp = np.array([np.mean(entropy_adn_up[entropy_adn_up[:,0]==i,1]) for i in range(20)])
# tmp2 = np.array([np.mean(entropy_lmn_up[entropy_lmn_up[:,0]==i,1]) for i in range(20)])

# tmp = tmp - tmp.mean()
# tmp = tmp / tmp.std()

# tmp2 = tmp2 - tmp2.mean()
# tmp2 = tmp2 - tmp2.std()

# fig = figure()
# ax = fig.add_subplot(111)
# ax.plot(tmp, label = 'adn', color = 'red')
# ax2 = ax.twinx()
# ax2.plot(tmp2, label = 'lmn')
# fig.legend()

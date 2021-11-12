import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pingouin import partial_corr

def zscore_rate(rate):
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	return rate



data_directory = '/mnt/Data2/Opto/A8000/A8015/A8015-210825A'

episodes = ['sleep', 'wake', 'sleep', 'wake']
# events = ['1', '3']

# episodes = ['sleep', 'wake', 'sleep']
events = ['1', '3']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)


position 							= loadPosition(data_directory, events, episodes, 2, 1)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
sws_ep								= loadEpoch(data_directory, 'sws')
acceleration						= loadAuxiliary(data_directory)
# #sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

#################
# TUNING CURVES
tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60)
#tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tuning_curves 						= smoothAngularTuningCurves(tuning_curves, 10, 2)

tokeep, stat 						= findHDCells(tuning_curves, z=10, p = 0.001)
#tokeep = list(spikes.keys())
#tokeep = [0, 2, 4, 5]

tcurves 							= tuning_curves[tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]


#################
# OPTO
opto_ep 							= loadOptoEp(data_directory, epoch=0, n_channels = 2, channel = 0)
opto_ep 							= opto_ep.merge_close_intervals(40000)
frates, rasters, bins, stim_duration = computeRasterOpto(spikes, opto_ep, 1000)

opto_ep = sws_ep.intersect(opto_ep)

nopto_ep = nts.IntervalSet(
	start = opto_ep['start'] - (opto_ep['end'] - opto_ep['start']),
	end = opto_ep['start']
	)



###################
# PARTIAL CORRELATION
def computePartialCorrelation(x, y, z):	
	df = pd.DataFrame(data = np.vstack([x,y,z]).T, columns = ['x', 'y', 'z'])
	pc = partial_corr(data = df, x = 'x', y = 'y', covar = 'z').round(3)
	return pc['r'].values[0]

def MakePartialCorrelation(spikes, ep1, ep2, bin1, bin2, std1=3, std2=3):
	from pingouin import partial_corr
	rate1 = zscore_rate(binSpikeTrain(spikes, ep1, bin1, std1))
	rate2 = zscore_rate(binSpikeTrain(spikes, ep2, bin2, std2))
	mua1 = zscore_rate(binSpikeTrain({0:pd.concat(spikes.values())}, ep1, bin1, std1))
	mua2 = zscore_rate(binSpikeTrain({0:pd.concat(spikes.values())}, ep2, bin2, std2))
	tokeep = list(spikes.keys())
	r1 = [computePartialCorrelation(rate1[:,i], rate1[:,j], mua1[:,0]) for i,j in combinations(range(len(tokeep)), 2)]
	r2 = [computePartialCorrelation(rate2[:,i], rate2[:,j], mua2[:,0]) for i,j in combinations(range(len(tokeep)), 2)]
	r = pd.DataFrame(data = np.vstack((r1, r2)).T)
	pairs = list(combinations(tokeep, 2))
	r.index = pd.Index(pairs)
	return r

wak_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, wake_ep, 300, 3))
nopto_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, nopto_ep, 15, 3))
opto_rate = zscore_rate(binSpikeTrain({n:spikes[n] for n in tokeep}, opto_ep, 15, 3))

wak_mua = zscore_rate(binSpikeTrain({0:pd.concat({n:spikes[n] for n in tokeep}.values())}, wake_ep, 300, 3))
nopto_mua = zscore_rate(binSpikeTrain({0:pd.concat({n:spikes[n] for n in tokeep}.values())}, nopto_ep, 15, 3))
opto_mua = zscore_rate(binSpikeTrain({0:pd.concat({n:spikes[n] for n in tokeep}.values())}, opto_ep, 15, 3))


def computePartialCorrelation(x, y, z):	
	df = pd.DataFrame(data = np.vstack([x,y,z]).T, columns = ['x', 'y', 'z'])
	pc = partial_corr(data = df, x = 'x', y = 'y', covar = 'z').round(3)
	return pc['r'].values[0]


r_wak = [computePartialCorrelation(wak_rate[:,i], wak_rate[:,j], wak_mua[:,0]) for i,j in combinations(range(len(tokeep)), 2)]
r_nopto = [computePartialCorrelation(nopto_rate[:,i], nopto_rate[:,j], nopto_mua[:,0]) for i,j in combinations(range(len(tokeep)), 2)]
r_opto = [computePartialCorrelation(opto_rate[:,i], opto_rate[:,j], opto_mua[:,0]) for i,j in combinations(range(len(tokeep)), 2)]

r = pd.DataFrame(data = np.vstack((r_wak, r_nopto, r_opto)).T)

pairs = list(combinations(tokeep, 2))

r.index = pd.Index(pairs)
r.columns = pd.Index(['wak', 'nopto', 'opto'])



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue']
shank = shank.flatten()


figure()
count = 1
for j in np.unique(shank):
	neurons = np.where(shank == j)[0]
	for k,i in enumerate(neurons):
		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1])
		# plot(tuning_curves2[1][i], '--', color = colors[shank[i]-1])
		if i in tokeep:
			plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i), color = colors[shank[i]-1], linewidth = 3)
		# legend()
		count+=1
		gca().set_xticklabels([])


figure()
subplot(121)
plot(r['wak'], r['opto'], 'o', color = 'red', alpha = 0.5)
m, b = np.polyfit(r['wak'].values, r['opto'].values, 1)
x = np.linspace(r['wak'].min(), r['wak'].max(),5)
plot(x, x*m + b, label = 'opto r='+str(np.round(m,3)), color = 'red')
xlabel('wake')
plot(r['wak'], r['nopto'], 'o', color = 'grey', alpha = 0.5)
m, b = np.polyfit(r['wak'].values, r['nopto'].values, 1)
x = np.linspace(r['wak'].min(), r['wak'].max(),5)
plot(x, x*m + b, label = 'non-opto r='+str(np.round(m,3)), color = 'grey')
legend()
subplot(122)

[plot([0,1],r.loc[p,['nopto', 'opto']], 'o-') for p in r.index]
xticks([0,1], ['sws', 'opto'])

legend()


figure()

for n in peaks.index.values:
	plot(spikes[n].restrict(opto_ep).fillna(peaks.loc[n]), '|', color = 'red')
for n in peaks.index.values:
	plot(spikes[n].restrict(nopto_ep).fillna(peaks.loc[n]), '|', color = 'grey')

[axvspan(opto_ep.loc[i,'start'],opto_ep.loc[i,'end'], alpha = 0.1, color = 'red') for i in opto_ep.index]


show()
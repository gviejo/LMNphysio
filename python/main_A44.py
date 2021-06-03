import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean


data_directory = '/media/guillaume/Seagate/Kraken/K2/A6403/A6403-201228'

episodes = ['sleep', 'wake']
# events = ['1', '3']

# episodes = ['sleep', 'wake', 'sleep']
events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)

position 							= loadPosition(data_directory, events, episodes, 2, 0)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
# acceleration						= loadAuxiliary(data_directory)
# sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

#################
# TUNING CURVES
tuning_curves 						= computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves						= smoothAngularTuningCurves(tuning_curves, 10, 2)

tokeep, stat 						= findHDCells(tuning_curves, z=10, p = 0.001)

tcurves 							= tuning_curves[tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]

figure()
for i in tuning_curves.columns:
	subplot(int(np.ceil(np.sqrt(tuning_curves.shape[1]))),int(np.ceil(np.sqrt(tuning_curves.shape[1]))),i+1, projection='polar')
	plot(tuning_curves[i])

tcurves = tuning_curves


#################
# OPTO
opto_ep = loadOptoEp(data_directory, 1, 2, 1)
opto_ep2 = opto_ep.merge_close_intervals(1e6)

evt_file = os.path.join(data_directory, 'opto_ep.evt.py.opt')
writeNeuroscopeEvents(evt_file, opto_ep, 'Opto')

start = []
end = []
for i in opto_ep2.index.values:
	start.append(int(opto_ep2.loc[i,'start']-opto_ep2.loc[[i]].tot_length()))
	end.append(opto_ep2.loc[i,'start'])

nonopto_ep = nts.IntervalSet(start = start, end = end)


frates, rasters = computeRasterOpto(spikes, opto_ep2, 1000)
frates2, rasters2 = computeRasterOpto(spikes, nonopto_ep, 1000)


tc_opto 		= computeAngularTuningCurves(spikes, position['ry'], opto_ep2, 61)
tc_nonopto		= computeAngularTuningCurves(spikes, position['ry'], nonopto_ep, 61)
tc_opto			= smoothAngularTuningCurves(tc_opto, 10, 2)
tc_nonopto		= smoothAngularTuningCurves(tc_nonopto, 10, 2)

############################################################################################### 
# PLOT
###############################################################################################

fig = figure()
for i, n in enumerate(tcurves.columns):
	subplot(4, 4, i+1, projection = 'polar')
	plot(tcurves[n])

fig = figure()
for i, n in enumerate(tc_opto.columns):
	subplot(4, 4, i+1, projection = 'polar')	
	plot(tc_opto[n], label = 'opto')	
	plot(tc_nonopto[n], label = 'non-opto')
	if i == 0: legend()



from matplotlib import gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec

start = 120000
stop = 240000
fig = figure()
# gs = gridspec.GridSpec(3,4)
subplot(3,4,1)
plot(frates)
plot(frates.mean(1), linewidth = 5)
axvline(start)
axvline(stop)
for i, n in enumerate(rasters.keys()):
	subgs = GridSpecFromSubplotSpec(2, 1, subplot(3,4,i+2))
	subplot(subgs[0,0])	
	plot(frates[n])
	plot(frates2[n])
	axvline(start)
	axvline(stop)
	subplot(subgs[1,0])
	plot(rasters[n], '.', markersize = 2)
	axvline(start*1000)
	axvline(stop*1000)




sys.exit()

figure()
plot(cc_slp.mean(1), label = 'sleep')
plot(cc_opt.mean(1), label = 'opto')
legend()

figure()
plot(cc_slp[pairs.index.values[27:]].mean(1), label = 'sleep')
plot(cc_opt[pairs.index.values[27:]].mean(1), label = 'opto')
legend()

figure()
plot(cc_slp[pairs.index.values[:27]].mean(1), label = 'sleep')
plot(cc_opt[pairs.index.values[:27]].mean(1), label = 'opto')
legend()

figure()
subplot(121)
plot(cc_slp[pairs.index.values[:27]].mean(1), label = 'sleep_bottom', color = 'blue')
plot(cc_slp[pairs.index.values[27:]].mean(1), label = 'sleep_top', color = 'blue')
legend()
subplot(122)
plot(cc_opt[pairs.index.values[:27]].mean(1), label = 'opto_bottom', color = 'orange')
plot(cc_opt[pairs.index.values[27:]].mean(1), label = 'opto_top', color = 'orange')
legend()


figure()
plot(cc_slp[pairs.index.values[27:]].mean(1), label = 'sleep_bottom', color = 'blue')
plot(cc_opt[pairs.index.values[27:]].mean(1), label = 'opto_bottom', color = 'orange')

opto_ep1 = opto_ep.loc[0:10]
opto_ep2 = opto_ep.loc[10:]

nonopto_ep1 = nonopto_ep.loc[0:10]
nonopto_ep2 = nonopto_ep.loc[10:]

cc_optc = {}
cc_slpc = {}

for i, ep in enumerate([opto_ep1, opto_ep2]):
	tmp = compute_CrossCorrs(spikes, ep, 2, 2000, norm=True)
	tmp = tmp.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)
	cc_optc[i] = tmp

for i, ep in enumerate([nonopto_ep1, nonopto_ep2]):
	tmp = compute_CrossCorrs(spikes, ep, 2, 2000, norm=True)
	tmp = tmp.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)
	cc_slpc[i] = tmp

figure()
subplot(221)
plot(cc_slpc[0][pairs.index.values[:27]].mean(1), label = 'sleep_top', color = 'blue')
plot(cc_slpc[1][pairs.index.values[:27]].mean(1), label = 'sleep_top', color = 'blue')
# plot(cc_slpc[0][pairs.index.values[27:]].mean(1), label = 'sleep_top', color = 'blue')
# plot(cc_slpc[1][pairs.index.values[27:]].mean(1), label = 'sleep_top', color = 'blue')
legend()
subplot(222)
plot(cc_optc[0][pairs.index.values[:27]].mean(1), label = 'opto_top', color = 'orange')
plot(cc_optc[1][pairs.index.values[:27]].mean(1), label = 'opto_top', color = 'orange')

subplot(223)
plot(cc_slpc[0][pairs.index.values[27:]].mean(1), label = 'sleep_bottom', color = 'blue')
plot(cc_slpc[1][pairs.index.values[27:]].mean(1), label = 'sleep_bottom', color = 'blue')
# plot(cc_slpc[0][pairs.index.values[27:]].mean(1), label = 'sleep_top', color = 'blue')
# plot(cc_slpc[1][pairs.index.values[27:]].mean(1), label = 'sleep_top', color = 'blue')
legend()
subplot(224)
plot(cc_optc[0][pairs.index.values[27:]].mean(1), label = 'opto_bottom', color = 'orange')
plot(cc_optc[1][pairs.index.values[27:]].mean(1), label = 'opto_bottom', color = 'orange')


legend()


figure()
subplot(221)
plot(cc_slpc[0][pairs.index.values[:27]].mean(1), label = 'sleep_bottom', color = 'blue')
plot(cc_slpc[0][pairs.index.values[27:]].mean(1), label = 'sleep_top', color = 'blue')
legend()
subplot(222)
plot(cc_optc[0][pairs.index.values[:27]].mean(1), label = 'opto_bottom', color = 'orange')
plot(cc_optc[0][pairs.index.values[27:]].mean(1), label = 'opto_top', color = 'orange')
legend()
subplot(223)
plot(cc_slpc[1][pairs.index.values[:27]].mean(1), label = 'sleep_bottom', color = 'blue')
plot(cc_slpc[1][pairs.index.values[27:]].mean(1), label = 'sleep_top', color = 'blue')
legend()
subplot(224)
plot(cc_optc[1][pairs.index.values[:27]].mean(1), label = 'opto_bottom', color = 'orange')
plot(cc_optc[1][pairs.index.values[27:]].mean(1), label = 'opto_top', color = 'orange')
legend()



diff_cc_slp = []
diff_cc_opt = []

for i in opto_ep.index.values:
	tmp_slp = compute_CrossCorrs(spikes, nonopto_ep.loc[[i]], 2, 2000, norm=True)
	tmp_opt = compute_CrossCorrs(spikes, opto_ep.loc[[i]], 2, 2000, norm=True)

	# tmp_slp = pd.concat([tmp_slp.loc[:-250], tmp_slp.loc[250:]])
	# tmp_opt = pd.concat([tmp_opt.loc[:-250], tmp_opt.loc[250:]])	
	# tmp_slp.index = np.arange(len(tmp_slp.index))
	# tmp_opt.index = np.arange(len(tmp_opt.index))

	tmp_slp = tmp_slp.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)
	tmp_opt = tmp_opt.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)

	d_slp = tmp_slp[pairs.index.values[0:27]].mean(1) - tmp_slp[pairs.index.values[27:]].mean(1)
	d_opt = tmp_opt[pairs.index.values[0:27]].mean(1) - tmp_opt[pairs.index.values[27:]].mean(1)

	# d_slp = d_slp.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)
	# d_opt = d_opt.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)

	d_slp = pd.concat([d_slp.loc[:-250], d_slp.loc[250:]])
	d_opt = pd.concat([d_opt.loc[:-250], d_opt.loc[250:]])

	diff_cc_slp.append(d_slp)
	diff_cc_opt.append(d_opt)

diff_cc_slp = pd.concat(diff_cc_slp, 1)
diff_cc_opt = pd.concat(diff_cc_opt, 1)

diff_cc_slp.index = np.arange(len(diff_cc_slp.index))
diff_cc_opt.index = np.arange(len(diff_cc_opt.index))

diff_cc_slp = diff_cc_slp.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)
diff_cc_opt = diff_cc_opt.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 4.0)


d_cc_slp = diff_cc_slp.values
d_cc_opt = diff_cc_opt.values

# d_cc_slp = d_cc_slp - d_cc_slp.mean(0)
# d_cc_opt = d_cc_opt - d_cc_opt.mean(0)

# d_cc_slp = d_cc_slp / d_cc_slp.std(0)
# d_cc_opt = d_cc_opt / d_cc_opt.std(0)


figure()
subplot(121)
plot(diff_cc_slp)
title('sleep')
subplot(122)
plot(diff_cc_opt)
title('opto')

figure()
subplot(121)
imshow(d_cc_slp.T, aspect = 'auto')
title('sleep')
subplot(122)
imshow(d_cc_opt.T, aspect = 'auto')
title('opto')

figure()
plot(d_cc_slp.sum(0), label = 'sleep')
plot(d_cc_opt.sum(0), label = 'opto')
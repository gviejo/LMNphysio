# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-08-10 17:16:25
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-05-09 14:32:53
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
try:
    from functions import *
except:
    sys.path.append(os.path.expanduser("~/LMNphysio/pyna"))
    from functions import *

from itertools import combinations, product
from matplotlib.pyplot import *


if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'

# path = '/mnt/DataRAID2/LMN-ADN/A5043/A5043-230301A'
path = os.path.join(data_directory, "LMN-PSB/A3019/A3019-220701A")



# data = nap.load_session(path, 'neurosuite')
basename = os.path.basename(path)
# filepath = os.path.join(path, "pynapplenwb", basename + ".nwb") 
filepath = os.path.join(path, "kilosort4_bk", basename + ".nwb")

nwb = nap.load_file(filepath)

spikes = nwb['units']
# spikes = spikes.getby_threshold("rate", 1)

position = []
columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
for k in columns:
    position.append(nwb[k].values)
position = np.array(position)
position = np.transpose(position)
position = nap.TsdFrame(
    t=nwb['x'].t,
    d=position,
    columns=columns,
    time_support=nwb['position_time_support'])

epochs = nwb['epochs']
wake_ep = epochs[epochs.tags == "wake"]
wake_ep = nwb['position_time_support']
sws_ep = nwb['sws']
rem_ep = nwb['rem']

angle = position['ry']

nwb.close()

up_ep = read_neuroscope_intervals(path, basename, 'up')
down_ep = read_neuroscope_intervals(path, basename, 'down')

down_ep = down_ep.drop_short_intervals(0.02).drop_long_intervals(0.7)


# waveforms = nwb.nwb.units.to_dataframe()['waveform_mean']
# waveforms = np.array([waveforms[i] for i in waveforms.keys()])

# turning by pi
tmp = np.unwrap(position['ry'].values)
tmp += np.pi
tmp = np.mod(tmp, 2*np.pi)
angle = nap.Tsd(t = position.index.values, d = tmp, time_support=position.time_support)


dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
spikes.set_info(halfr = r)

psb2 = spikes.getby_category("location")['psb'].index
psb = spikes.getby_category("location")['psb'].getby_threshold('SI', 0.1).getby_threshold('halfr', 0.5).index
lmn = spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.1).getby_threshold('halfr', 0.5).index

tokeep = list(psb) + list(lmn)

spikes2 = spikes[tokeep]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = spikes.metadata.group.values


# cc = nap.compute_crosscorrelogram((spikes[lmn], spikes[psb]), 0.001, 0.5, sws_ep, norm=True)


# ahv = np.gradient(np.unwrap(angle).bin_average(0.05))/np.mean(np.diff(angle.t))

# tcahv = nap.compute_1d_tuning_curves(spikes, ahv, 100, wake_ep, minmax=(-50, 50))



figure()
count = 1
for l,j in enumerate(np.unique(shank)):
    neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
    for k,i in enumerate(neurons):      
        subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')    
        plot(tuning_curves[i], label = str(np.round(SI.loc[i].values[0], 4)), color = colors[l])
        title(i)
        legend()
        count+=1
        gca().set_xticklabels([])
        if i in tokeep:
            fill_between(tuning_curves[i].index.values,
                np.zeros_like(tuning_curves[i].values),
                tuning_curves[i].values, 
                color = colors[l])


#############################
# RASTER
#############################

tcurves = tuning_curves

# peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
peaks = tcurves.idxmax()

psb = peaks[psb].sort_values().index.values
lmn = peaks[lmn].sort_values().index.values

# sys.exit()


sws2_ep = sws_ep[np.argsort(sws_ep.end-sws_ep.start)[-2]]


# psb_idx = [psb[12], psb[13], psb[4]]
# lmn_idx = [lmn[9], lmn[11], lmn[3]]

exs = { 'wak':nap.IntervalSet(start = 9968.5, end = 9987, time_units='s'),
        'rem':nap.IntervalSet(start = 13383.819, end= 13390, time_units = 's'),
        #'sws':nap.IntervalSet(start = 6555.6578, end = 6557.0760, time_units = 's')}
        #'sws':nap.IntervalSet(start = 5318.6593, end = 5320.0163, time_units = 's')
        'sws':nap.IntervalSet(start = 5897.10, end = 5898.45, time_units = 's'),
        # 'sws':nap.IntervalSet(start = 5895.30, end = 5898.45, time_units = 's')
        'nrem2':nap.IntervalSet(start = 5800.71, end = 5805.2, time_units = 's'),
        'nrem3':nap.IntervalSet(start = 5808.5, end = 5812.7, time_units = 's')
        }



# wake
figure()
ax = subplot(313)
plot(angle)
title("wake")
# plot(angle_wak, '--')
subplot(311, sharex = ax)
title("wake")
for i,n in enumerate(psb):
    plot(spikes[n].restrict(wake_ep).fillna(i), '|', 
        markersize = 10, color='grey', alpha=0.5)

axvspan(exs['wak'].start[0], exs['wak'].end[0], alpha=0.5)

subplot(312, sharex = ax)
for i,n in enumerate(lmn):
    plot(spikes[n].restrict(wake_ep).fillna(i), '|', 
        markersize = 10, color='grey', alpha=0.5)

axvspan(exs['wak'].start[0], exs['wak'].end[0], alpha=0.5)



# sys.exit()
# sws_ep = sws_ep[np.argmax(sws_ep.end-sws_ep.start)]



mua = spikes[psb2].to_tsd().count(0.01, sws_ep)
mua = mua.smooth(0.02, size_factor=20)

tmua = (mua - mua.mean())/mua.std()

down_ep = tmua.threshold(np.percentile(tmua.values, 20), 'below').time_support.drop_short_intervals(0.02).drop_long_intervals(0.7).merge_close_intervals(0.02)
up_ep = sws_ep.set_diff(down_ep)

up_ep = up_ep.intersect(sws_ep[13])
down_ep = down_ep.intersect(sws_ep[13])


# sws
figure()
ax = subplot(311)
plot(mua, '-', color='grey')
plot(mua.restrict(up_ep), 'o', color='red')
plot(mua.restrict(down_ep), 'o', color='green')
for s, e in down_ep.values:
    axvspan(s, e, color='green', alpha=0.2)
for s, e in up_ep.values:
    axvspan(s, e, color='red', alpha=0.2)


title("sws")

subplot(312, sharex=ax)
title("sws")
for s, e in down_ep.values:
    axvspan(s, e, color='green', alpha=0.2)

for i,n in enumerate(psb2):
    plot(spikes[n].restrict(up_ep).fillna(i), '|', 
        markersize = 10, color='red', alpha=1, mew=2)
    plot(spikes[n].restrict(down_ep).fillna(i), '|', 
        markersize = 10, color='green', alpha=1, mew=2)


axvspan(5800.71, 5805.2, color='blue', alpha=0.1)
# axvspan(5808.5, 5812.7, color='red', alpha=0.01)

# axvspan(exs['sws'].start[0], exs['sws'].end[0], alpha=0.5)

subplot(313, sharex = ax)
for s, e in down_ep.values:
    axvspan(s, e, color='green', alpha=0.2)

for i,n in enumerate(lmn):
    plot(spikes[n].restrict(up_ep).fillna(i), '|', 
        markersize = 10, color='red', alpha=1, mew=2)
    plot(spikes[n].restrict(down_ep).fillna(i), '|', 
        markersize = 10, color='green', alpha=1, mew=2)


# axvspan(5800.71, 5805.2, color='red', alpha=0.01)
# axvspan(5808.5, 5812.7, color='red', alpha=0.01)

# axvspan(exs['sws'].start[0], exs['sws'].end[0], alpha=0.5)

show()




###########################################################################
#SAVING
###########################################################################

datatosave = { #'wak':angle_wak,
              #'rem':angle_rem,
              #'sws':angle_sws,
              'tcurves':tcurves,
              'angle':angle,
              'peaks':peaks,
              'spikes':spikes,
              'psb':psb,
              'psb2':psb2,
              'lmn':lmn,
              # 'p_rem':proba_rem,
              # 'p_sws':proba_sws,
              # 'p_wak':proba_angle_wak,
              'up_ep':up_ep,
              'down_ep':down_ep,
              'tokeep':np.array(tokeep),
              # 'ex_sws':exs[os.path.basename(path)]['sws'],
              # 'ex_rem':exs[os.path.basename(path)]['rem'],
              # 'ex_wak':exs[os.path.basename(path)]['wak'],
              # 'waveforms':waveforms
          }

import _pickle as cPickle
filepath = os.path.join(os.path.expanduser("~"), f'Dropbox/LMNphysio/data/DATA_FIG_LMN_PSB_{basename}.pickle')
cPickle.dump(datatosave, open(filepath, 'wb'))

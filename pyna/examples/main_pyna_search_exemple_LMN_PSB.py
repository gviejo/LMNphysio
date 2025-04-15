# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-08-10 17:16:25
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-04-15 16:33:29
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
filepath = os.path.join(path, "pynapplenwb", basename + ".nwb")
# filepath = os.path.join(path, "kilosort4", basename + ".nwb")

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

# waveforms = nwb.nwb.units.to_dataframe()['waveform_mean']
# waveforms = np.array([waveforms[i] for i in waveforms.keys()])

# # turning by pi
# tmp = np.unwrap(position['ry'].values)
# tmp += np.pi
# tmp = np.mod(tmp, 2*np.pi)
# angle = nap.Tsd(t = position.index.values, d = tmp, time_support=position.time_support)


dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
spikes.set_info(halfr = r)


psb = spikes.getby_category("location")['psb'].getby_threshold('SI', 0.2).getby_threshold('halfr', 0.5).index
lmn = spikes.getby_category("location")['lmn'].getby_threshold('SI', 0.1).getby_threshold('halfr', 0.5).index

tokeep = list(psb) + list(lmn)

spikes2 = spikes[tokeep]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'wheat', 'indianred', 'royalblue', 'plum', 'forestgreen']

shank = spikes._metadata.group.values


cc = nap.compute_crosscorrelogram((spikes[lmn], spikes[psb]), 0.001, 0.5, sws_ep, norm=True)


ahv = np.gradient(np.unwrap(angle).bin_average(0.05))/np.mean(np.diff(angle.t))

tcahv = nap.compute_1d_tuning_curves(spikes, ahv, 100, wake_ep, minmax=(-50, 50))



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





cc2 = cc[cc.columns[cc.loc[-0.1:0.1].idxmax()<0]]

figure()
for i, p in enumerate(cc2.max().sort_values().index.values[-50:]):
    subplot(5,10,i+1)
    plot(cc2[p])
    title(p)


ps = [(89,39), (73, 39), (84,39), (78,39), (78, 28), (74,37), (69,37), (91, 28)]

for p in ps:
    figure()
    # p = (78, 39)
    subplot(131)
    plot(cc[p])
    subplot(132, projection='polar')
    plot(tuning_curves[p[0]], label = 'lmn')
    plot(tuning_curves[p[1]], label = 'psb')
    legend()
    subplot(133)
    plot(tcahv[p[0]], label = 'lmn')
    plot(tcahv[p[1]], label = 'psb')
    legend()

show()

#############################
# RASTER
#############################

tcurves = tuning_curves

peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

psb = peaks[psb].sort_values().index.values
lmn = peaks[lmn].sort_values().index.values

# sys.exit()


sws2_ep = sws_ep[np.argsort(sws_ep.end-sws_ep.start)[-2]]


# psb_idx = [psb[12], psb[13], psb[4]]
# lmn_idx = [lmn[9], lmn[11], lmn[3]]

# wake
figure()
ax = subplot(313)
plot(angle)
title("wake")
# plot(angle_wak, '--')
subplot(311, sharex = ax)
for i,n in enumerate(psb):
    plot(spikes[n].restrict(wake_ep).fillna(i), '|', 
        markersize = 10, color='grey', alpha=0.5)
    if n == 39:
        plot(spikes[n].restrict(wake_ep).fillna(i), '|', 
            markersize = 10, color='red')

subplot(312, sharex = ax)
for i,n in enumerate(lmn):
    plot(spikes[n].restrict(wake_ep).fillna(i), '|', 
        markersize = 10, color='grey', alpha=0.5)

    if n == 101:
        plot(spikes[n].restrict(wake_ep).fillna(i), '|', 
            markersize = 10, color='red')


# [axvline(a) for a in exs[basename]['wak'].values[0]]

# sws
figure()
ax = subplot(211)
title("sws")
for i,n in enumerate(psb):
    plot(spikes[n].restrict(sws_ep).fillna(i), '|', 
        markersize = 10, color='grey', alpha=1)

    # if n == 39:
    #     plot(spikes[n].restrict(sws_ep).fillna(i), '|', 
    #         markersize = 10, color='red')
    axvspan(5800.71, 5805.2, color='red', alpha=0.01)
    axvspan(5808.5, 5812.7, color='red', alpha=0.01)


subplot(212, sharex = ax)
for i,n in enumerate(lmn):
    plot(spikes[n].restrict(sws_ep).fillna(i), '|', 
        markersize = 15, color='grey', alpha=1)

    # if n == 101:
    #     plot(spikes[n].restrict(sws_ep).fillna(i), '|', 
    #         markersize = 10, color='red')

    axvspan(5800.71, 5805.2, color='red', alpha=0.01)
    axvspan(5808.5, 5812.7, color='red', alpha=0.01)


show()




# # rem
# figure()
# ax = subplot(311)
# plot(angle_rem, '--')
# title("rem")
# subplot(312, sharex = ax)
# for i,n in enumerate(psb):
#     plot(spikes[n].restrict(rem_ep).fillna(i), '|', markersize = 10)
# subplot(313, sharex = ax)
# for i,n in enumerate(lmn):
#     plot(spikes[n].restrict(rem_ep).fillna(i), '|', markersize = 10)

# show()



###########################################################################
#SAVING
###########################################################################
# A5011-201014
# exs = {"A5011-201014A":
#             { 'wak':nap.IntervalSet(start = 9604.5, end = 9613.7, time_units='s'),
#             'rem':nap.IntervalSet(start = 15710.150000, end= 15720.363258, time_units = 's'),
#             'sws':nap.IntervalSet(start = 13653.07, end=13654.86, time_units = 's')},
#         "A5043-230301A":
#             { 'wak':nap.IntervalSet(start = 4560.50, end = 4600.00),
#             'rem':nap.IntervalSet(start = 7600, end= 7627.0, time_units = 's'),
#             'sws':nap.IntervalSet(start = 1522.73, end = 1530.38)}
#     }


# datatosave = { 'wak':angle_wak,
#               'rem':angle_rem,
#               'sws':angle_sws,
#               'tcurves':tcurves,
#               'angle':angle,
#               'peaks':peaks,
#               'spikes':spikes,
#               'psb':psb,
#               'lmn':lmn,
#               'p_rem':proba_rem,
#               'p_sws':proba_sws,
#               'p_wak':proba_angle_wak,
#               # 'up_ep':up_ep,
#               # 'down_ep':down_ep,
#               'tokeep':tokeep,
#               'ex_sws':exs[os.path.basename(path)]['sws'],
#               'ex_rem':exs[os.path.basename(path)]['rem'],
#               'ex_wak':exs[os.path.basename(path)]['wak'],
#               'waveforms':waveforms
#           }

# import _pickle as cPickle
# # cPickle.dump(datatosave, open('../figures/figures_adrien_2022/fig_1_decoding.pickle', 'wb'))


# filepath = os.path.join(os.path.expanduser("~"), 'Dropbox/LMNphysio/data/DATA_FIG_LMN_ADN_{}.pickle'.format(os.path.basename(path)))

# cPickle.dump(datatosave, open(filepath, 'wb'))

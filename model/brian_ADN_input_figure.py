# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-02-21 12:10:37
# @Last Modified by:   gviejo
# @Last Modified time: 2025-06-14 20:50:07

import scipy.io
import sys,os
import numpy as np
from brian2 import *
import pandas as pd
import pynapple as nap
from functions import smoothAngularTuningCurves
import sys
from itertools import combinations, product

	

spk_times = []
start = 0
firing_rates = np.arange(1, 140, 5)

epochs = []

for i, fr in enumerate(firing_rates):
	tmp = np.arange(0, 60, 1/fr)
	tmp = tmp + start
	spk_times.append(tmp)
	epochs.append([start, start+60])
	start += 60

spk_times = np.hstack(spk_times)
spk_times = np.unique(spk_times)

epochs = np.array(epochs)


#############################################################################
# BRIAN NETWORK
#############################################################################
set_device('cpp_standalone')
start_scope()


duration = (60*len(firing_rates)) * second
tau_adn = 100 * ms
tau_lmn = 100 * ms


dt_inp = 0.1 * ms


indices = np.zeros_like(spk_times)
spk_times = spk_times * second

INP_group = SpikeGeneratorGroup(1, indices, spk_times, dt=dt_inp)




def sigmoide(I, b=100.0):
	return 10/(1+np.exp(-(I-0.75)*b))


eqs_adn = '''
dv/dt = ((10/(1+exp(-(I-0.75)*60)))-v)/tau_adn : 1
dI/dt = -I/tau_adn : 1
'''
eqs_lmn 	= '''
dv/dt = -v/tau_lmn : 1
'''


ADN_group = NeuronGroup(1, model=eqs_adn, threshold='v>0.95', reset='v=0.0', method = 'euler', refractory=10*ms)
LMN_group = NeuronGroup(1, model=eqs_lmn, threshold='v>1', reset='v=0.0', method = 'euler', refractory=10*ms)


INP_to_ADN = Synapses(INP_group, ADN_group, 'w : 1', on_pre='I += w')
INP_to_ADN.connect(i = 0, j = 0)
INP_to_ADN.w = 0.15


INP_to_LMN = Synapses(INP_group, LMN_group, 'w : 1', on_pre='v += w')
INP_to_LMN.connect(i = 0, j = 0)
INP_to_LMN.w = 2.0



# ###########################################################################################################
# # Spike monitor
# ###########################################################################################################
lmn_mon = SpikeMonitor(LMN_group)
adn_mon = SpikeMonitor(ADN_group)

trace_I = StateMonitor(ADN_group, 'I', record=[0])
trace_v = StateMonitor(ADN_group, 'v', record=[0])


###########################################################################################################
# RUN
###########################################################################################################
run(duration, report = 'text')


spikes = nap.TsGroup(
	{0:nap.Ts(t=np.array(adn_mon.t)),
	1:nap.Ts(t=np.array(lmn_mon.t)),
	})



IO_fr = pd.DataFrame(index = firing_rates, columns = ['adn', 'lmn'])


for j, f in enumerate(firing_rates):
	for i, n in enumerate(['adn', 'lmn']):
		spk = spikes.restrict(nap.IntervalSet(start=epochs[j,0], end=epochs[j,1]))
		IO_fr.loc[f] = spk.rate

IO_fr = IO_fr.fillna(0)


figure()
plot(IO_fr['adn'], 'o-', color = 'r', label = 'adn')
plot(IO_fr['lmn'], 'o-', color = 'y', label = 'lmn')
legend()




figure()
ax = subplot(311)
plot(lmn_mon.t, lmn_mon.i, '.y', label = 'lmn')
plot(adn_mon.t, adn_mon.i+1, '.r', label = 'adn')
legend()

subplot(312, sharex = ax)
plot(trace_v.t, trace_v[0].v, label = 'adn v')
legend()


subplot(313, sharex = ax)
plot(trace_I.t, trace_I[0].I, label = 'adn I')
plot(trace_I.t, sigmoide(trace_I[0].I), label = 'sigmoide(adn I)')
legend()

show()


path2 = '/home/guillaume/Dropbox/CosyneData'

IO_fr.to_hdf(path2 + '/IO_fr.hdf', 'fr')





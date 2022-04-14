# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-02-21 12:10:37
# @Last Modified by:   gviejo
# @Last Modified time: 2022-03-08 11:55:59

import scipy.io
import sys,os
import numpy as np
from brian2 import *
import pandas as pd
import pynapple as nap
from functions import smoothAngularTuningCurves
import sys
from itertools import combinations, product

	

set_device('cpp_standalone')
start_scope()


duration = 60 * second
tau_adn = 25 * ms
tau_inh = 10 * ms

dt_lmn = 1 * ms
n_lmn = 1#len(np.unique(spk_times.values))

indices = spk_times.values
spk_times = spk_times.index.values * ms

LMN_group = SpikeGeneratorGroup(n_lmn, indices, spk_times, dt=dt_lmn)

tau = 1 * ms
taul = 50 * ms



def sigmoide(v, b=100.0):
	return 1/(1+np.exp(-(v-0.5)*b))


eqs_adn = '''
dv/dt = ((1/(1+exp(-(I-0.5)*100)))-v)/tau : 1
dI/dt = -I/taul : 1
'''
# eqs_inh 	= '''
# dv/dt = - v/tau_inh : 1
# '''

ADN_group = NeuronGroup(n_lmn, model=eqs_adn, threshold='v>0.95', reset='v=0.0', method = 'euler', refractory=10*ms)

# INH_group = NeuronGroup(1, model = eqs_inh, threshold='v>1', reset='v=0', method = 'exact')

# ###########################################################################################################
# # SYNAPSES
# ###########################################################################################################
# LMN_to_ADN connection
LMN_to_ADN = Synapses(LMN_group, ADN_group, 'w : 1', on_pre='I += w')
LMN_to_ADN.connect(i = np.arange(n_lmn), j = np.arange(n_lmn))
LMN_to_ADN.w = 0.2



# ###########################################################################################################
# # Spike monitor
# ###########################################################################################################
lmn_mon = SpikeMonitor(LMN_group)
adn_mon = SpikeMonitor(ADN_group)
# # inh_mon = SpikeMonitor(INH_group)


trace = StateMonitor(ADN_group, 'v', record=[0])
trace2 = StateMonitor(ADN_group, 'I', record=[0])

###########################################################################################################
# RUN
###########################################################################################################
run(duration, report = 'text')


ax = subplot(311)
plot(trace.t, trace[0].v)
plot(lmn_mon.t, lmn_mon.i+1.1, '.y', label = 'lmn')
plot(adn_mon.t, adn_mon.i+1.2, '.r', label = 'adn')
xlabel('t')
ylabel('v')
subplot(312, sharex = ax)
plot(trace2.t, sigmoide(trace2[0].I))

subplot(313, sharex = ax)
plot(trace2.t, trace2[0].I)

show()



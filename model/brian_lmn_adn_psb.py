# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-11-17 22:50:30
# @Last Modified by:   gviejo
# @Last Modified time: 2022-11-17 23:57:00

import scipy.io
import sys,os
import numpy as np
from brian2 import *
import pandas as pd
import pynapple as nap
from functions import smoothAngularTuningCurves, getspikes_from_monitor
import sys
from itertools import combinations, product

#############################################################################
# BRIAN NETWORK
#############################################################################
set_device('cpp_standalone')
start_scope()


duration = 120 * second

ep = nap.IntervalSet(start = 0, end=duration)

tau_adn = 100 * ms
tau_lmn = 100 * ms
tau_psb = 100 * ms


n_neuron = 10


eqs_psb = '''
dv/dt = -v/ tau_psb : 1
'''
eqs_adn = '''
dv/dt = ((10/(1+exp(-(I-0.75)*60)))-v)/tau_adn : 1
dI/dt = -I/tau_adn : 1
'''
eqs_lmn = '''
dv/dt = -v/ tau_lmn : 1
'''
eqs_inh 	= '''
dv/dt = - v/tau_lmn : 1
'''

P = PoissonGroup(10, rates=np.random.uniform(1, 50, 10)*Hz)

LMN_group = NeuronGroup(n_neuron, model=eqs_lmn, threshold='v>0.95', reset='v=0.0', method = 'euler', refractory=10*ms)
ADN_group = NeuronGroup(n_neuron, model=eqs_adn, threshold='v>0.95', reset='v=0.0', method = 'euler', refractory=10*ms)
PSB_group = NeuronGroup(n_neuron, model=eqs_psb, threshold='v>0.95', reset='v=0.0', method = 'euler', refractory=10*ms)
# INT_group = NeuronGroup(n_lmn, model=eqs_inh, threshold='v>1', reset='v=0.0', method = 'euler', refractory=10*ms)

# ###########################################################################################################
# # SYNAPSES
# ###########################################################################################################
S = Synapses(P, LMN_group, 'w: 1', on_pre='v += w')
S.connect(condition='i!=j', p=0.5)
S.w = 0.1

# LMN to ADN connection
LMN_to_ADN = Synapses(LMN_group, ADN_group, 'w : 1', on_pre='I += w')
# LMN_to_ADN.connect(i = np.arange(n_lmn), j = np.arange(n_lmn))
LMN_to_ADN.connect(condition='i!=j', p=0.5)
LMN_to_ADN.w = 0.5

# # ADN to PSB connection
ADN_to_PSB = Synapses(ADN_group, PSB_group, 'w : 1', on_pre='v += w')
# ADN_to_PSB.connect(i = np.arange(n_neuron), j = np.arange(n_neuron))
ADN_to_PSB.connect(condition='i!=j', p=0.5) 
ADN_to_PSB.w = 0.5

# LMN to PSB connection
PSB_to_LMN = Synapses(PSB_group, LMN_group, 'w : 1', on_pre='v += w')
PSB_to_LMN.connect(condition='i!=j', p=0.5)
PSB_to_LMN.w = 0.1
PSB_to_PSB = Synapses(PSB_group, PSB_group, 'w : 1', on_pre='v += w')
PSB_to_PSB.connect(condition='i!=j', p=0.5) 
PSB_to_PSB.w = 0.1


# LMN_to_INT = Synapses(LMN_group, INT_group, 'w : 1', on_pre='v += w')
# LMN_to_INT.connect(i = np.arange(n_lmn), j = np.arange(n_lmn))
# LMN_to_INT.w = 2.0
# ADN_to_INH = Synapses(ADN_group, INH_group, 'w : 1', on_pre='v += w')
# ADN_to_INH.connect(i = np.arange(n_lmn), j = np.zeros(n_lmn, dtype=np.int))
# ADN_to_INH.w = 0.7
# INH_to_ADN = Synapses(INH_group, ADN_group, 'w : 1', on_pre='v -= w')
# INH_to_ADN.connect(i=np.zeros(n_lmn, dtype=np.int), j=np.arange(n_lmn))
# INH_to_ADN.w = 0.7



###########################################################################################################
# Spike monitor
###########################################################################################################
lmn_mon = SpikeMonitor(LMN_group)
adn_mon = SpikeMonitor(ADN_group)
psb_mon = SpikeMonitor(PSB_group)


###########################################################################################################
# RUN
###########################################################################################################
run(duration, report = 'text')


# ##########################################
# # ADN neurons
# ##########################################
adn_spikes = getspikes_from_monitor(adn_mon, len(ADN_group), ep)
lmn_spikes = getspikes_from_monitor(lmn_mon, len(LMN_group), ep)
psb_spikes = getspikes_from_monitor(psb_mon, len(PSB_group), ep)


figure()
for i, (spikes, st) in enumerate(zip([psb_spikes, adn_spikes, lmn_spikes], ['psb', 'adn', 'lmn'])):
	ax = subplot(3,1,i+1)
	[plot(spikes[n].fillna(n), '.g') for n in spikes.index]
	title(st)

show()
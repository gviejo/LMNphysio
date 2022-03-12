# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 15:17:30
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-11 10:45:17
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
import sys
from pycircstat.descriptive import mean as circmean
import sys
sys.path.append('../')
from functions import *



############################################################################################### 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/LMN-ADN/A5011/A5011-201014A'

data = nap.load_session(data_directory, 'neurosuite')

spikes = data.spikes#.getby_threshold('freq', 1.0)
angle = data.position['ry']
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')
rem_ep = data.read_neuroscope_intervals('rem')

# Only taking the first wake ep
wake_ep = wake_ep.loc[[0]]

adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI=SI)
spikes = spikes.getby_threshold('SI', 0.1, op = '>')
tuning_curves = tuning_curves[spikes.keys()]

tokeep = list(spikes.keys())

adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values

tcurves = tuning_curves

peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))


# Bayesian decoding
decoded_sleep, proba_angle_Sleep = nap.decode_1d(tuning_curves=tuning_curves,
                                                 group=spikes, 
                                                 ep=sws_ep,
                                                 bin_size=0.02, # second
                                                 feature=data.position['ry'], 
                                                 )

angle_sws = smoothAngle(decoded_sleep, 1)

peak = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

figure(figsize=(16,5))
# create a raster plot
ax = subplot(211)
for n in adn:
    plot(spikes[n].restrict(wake_ep).as_units('s').fillna(peaks[n]), '|')
    plot(angle)
subplot(212, sharex = ax)
for n in lmn:
    plot(spikes[n].restrict(wake_ep).as_units('s').fillna(peaks[n]), '|')
    plot(angle)
show()



sys.exit()

############################################################################
# SAVINGt
############################################################################
datatosave = {	'wak':angle_wak,
				'rem':angle_rem,
				'sws':angle_sws,
				'tcurves':tcurves,
				'angle':angle,
				'peaks':peaks,
			}

import _pickle as cPickle

cPickle.dump(datatosave, open('../../figures/figures_poster_2022/fig_cosyne_decoding.pickle', 'wb'))


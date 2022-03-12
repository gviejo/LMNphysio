# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 15:17:30
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-03 16:00:36
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

peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))


# DECODING XGB
# Binning Wake
bin_size_wake = 0.3
count = spikes.count(bin_size_wake, angle.time_support.loc[[0]])
count = count.as_dataframe()
ratewak = np.sqrt(count/bin_size_wake)
ratewak = ratewak.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
ratewak = zscore_rate(ratewak)
velocity = computeLinearVelocity(data.position[['x', 'z']], angle.time_support.loc[[0]], bin_size_wake)
newwake_ep = velocity.threshold(0.001).time_support	
ratewak = ratewak.restrict(newwake_ep)
angle2 = getBinnedAngle(angle, angle.time_support.loc[[0]], bin_size_wake).restrict(newwake_ep)
angle_wak, proba = xgb_decodage(Xr=ratewak, Yr=angle2, Xt=ratewak)

# Binning sws
bin_size_sws = 0.010
count = spikes.count(bin_size_sws, sws_ep)
count = count.as_dataframe()
ratesws = np.sqrt(count/bin_size_wake)
ratesws = ratesws.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
ratesws = zscore_rate(ratesws)
angle_sws, proba = xgb_decodage(Xr=ratewak, Yr=angle2, Xt=ratesws)
# angle_sws = angle_sws.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1)
angle_sws = nap.Tsd(angle_sws, time_support = sws_ep)

# Binning rem
bin_size_rem = 0.3
count = spikes.count(bin_size_rem, rem_ep)
count = count.as_dataframe()
raterem = np.sqrt(count/bin_size_rem)
raterem = raterem.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)
raterem = zscore_rate(raterem)
angle_rem, proba = xgb_decodage(Xr=ratewak, Yr=angle2, Xt=raterem)
# angle_rem = angle_rem.as_series().rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1)
angle_rem = nap.Tsd(angle_rem, time_support = rem_ep)



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


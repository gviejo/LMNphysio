# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-02-27 22:05:07
# @Last Modified by:   gviejo
# @Last Modified time: 2022-02-27 22:17:41
'''
    Minimal working example

'''

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

DATA_DIRECTORY = '/media/guillaume/LaCie/LMN-ADN/A5002/A5002-200303B'

# LOADING DATA
data = nap.load_session(DATA_DIRECTORY, 'neurosuite')

spikes = data.spikes
position = data.position
wake_ep = data.epochs['wake']

# spikes = spikes.getby_threshold('group', 2, '>=')
spikes = spikes.getby_category('group')[2]

# COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi))

SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], minmax=(0,2*np.pi))

sws_ep = data.read_neuroscope_intervals("sws")

# PLOT
plt.figure()
for i, n in enumerate(spikes.keys()):
    plt.subplot(5, 4, i+1, projection='polar')
    plt.plot(tuning_curves[n])
    plt.title(SI.loc[n].values)
    plt.xlabel([])
plt.show()

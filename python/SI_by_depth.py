import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
import os
from wrappers import *
import sys
from functions import *
from scipy.stats import norm
import scipy

data_directory = '/mnt/Data2/PSB/A8608/A8608-220106'


episodes = ['sleep', 'wake', 'wake', 'sleep', 'wake', 'wake', 'sleep']
events = ['1', '2', '4', '5']

#episodes = ['sleep', 'wake', 'wake', 'sleep', 'wake', 'wake', 'sleep']
#events = ['1', '2', '4', '5'] #which episodes are wake starting at 0
#episodes = ['sleep', 'wake', 'wake', 'wake', 'sleep', 'wake', 'wake', 'wake', 'sleep']
#events = ['1', '2', '3', '5', '6', '7']

spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
waveform, maxch = loadMeanWaveforms(data_directory)
#maxch numbering starts from top of shank to bottom

#For position: Adrian's rig is xzy by default, Guillaume's is yxz
position = loadPosition(data_directory, events, episodes)
wake_ep                             = loadEpoch(data_directory, 'wake', episodes)

tuning_curves = {}
spatial_info = {}
for i,n in enumerate(events):
    event_tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[i]], 60)
    event_tuning_curves = smoothAngularTuningCurves(event_tuning_curves, 10, 2)
    tuning_curves[i] = event_tuning_curves
    spatial_info[i] = computeSpatialInfo(event_tuning_curves, position['ry'], wake_ep.loc[[i]])

#################################################################################################################################################
#%%Plot
#################################################################################################################################################   

#x axis is depth, y axis is spatial information of the cell, coloured by shank
import matplotlib.pyplot as plt

figure()
plt.ylabel("Spatial Information")
plt.xlabel('Depth')
xAxis = [i + 0.5 for i in list(range(max(maxch)))]
colours = ["red", "turquoise"]
for n, cell in enumerate(spatial_info[0].index):
    x_axis = xAxis
    colouring = colours[shank[cell]-1]
    plt.scatter(maxch[cell], spatial_info[0].iloc[cell], alpha = 0.5, color = colouring, s = 20)
tight_layout()
show()


    
    
    

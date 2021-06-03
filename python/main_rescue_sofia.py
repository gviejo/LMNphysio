import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean


data_directory = '/mnt/DataGuillaume/PSB/A6207/A6207-210303'

episodes = ['sleep', 'wake']
# episodes = ['wake', 'sleep']
# episodes = ['wake']
events = ['1']
# events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position = loadPosition(data_directory, events, episodes, 2, 1)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory)


tuning_curves 						= computeAngularTuningCurves(spikes, position['rz'], wake_ep, 60)

tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)
		

############################################################################################### 
# PLOT
###############################################################################################


figure()
for i in spikes:
	subplot(6,7,i+1, projection = 'polar')
	plot(tuning_curves[i], label = str(shank[i]))
	legend()

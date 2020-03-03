import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

data_directory = '/home/guillaume/LMNphysio/data/A5000/A5001-200226A'


episodes = ['sleep', 'wake']
events = ['1']



spikes, shank 						= loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
position 							= loadPosition(data_directory, events, episodes)
wake_ep 							= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= loadEpoch(data_directory, 'sleep')					
acceleration						= loadAuxiliary(data_directory, n_probe = 2)
if 'A5002' in data_directory:
	acceleration 						= acceleration[[0,1,2]]
else:
	acceleration 						= acceleration[[3,4,5]]
acceleration.columns 				= pd.Index(np.arange(3))
sleep_ep 							= refineSleepFromAccel(acceleration, sleep_ep)

bins = np.arange(wake_ep.loc[0,'start'], wake_ep.loc[0,'end'], 5000)
spike_count = {}
for n in np.where(shank.flatten()==0)[0]:
	spike_count[n] = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = np.histogram(spikes[n].restrict(wake_ep).index.values, bins)[0])
spike_count = pd.DataFrame.from_dict(spike_count)

tmp = spike_count.mean(1)
tmp2 = tmp.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)


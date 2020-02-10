import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pychronux import mtspectrumpt

# sit = 0.5
# data = np.arange(1, 500+sit, sit)

data_directory 						= '../data/A1400/A1407/'
info 								= pd.read_csv(data_directory+'A1407.csv')
info 								= info.set_index('Session')
path 								= '../data/A1400/A1407/A1407-190416'
spikes, shank 						= loadSpikeData(path)
n_channels, fs, shank_to_channel 	= loadXML(path)
episodes 							= info.filter(like='Trial').loc[path.split("/")[-1]].dropna().values
events								= list(np.where(episodes == 'wake')[0].astype('str'))
position 							= loadPosition(path, events, episodes)
wake_ep 							= loadEpoch(path, 'wake', episodes)
sleep_ep 							= loadEpoch(path, 'sleep')					


data = spikes[0].restrict(wake_ep).as_units('s').index.values

Fs = 2000
fpass =  [1, 20]
tapers =  [2, 5]
trialave =  1
err =  [1, 0.05]
pad =  -1


S = mtspectrumpt(data, Fs, fpass, tapers)

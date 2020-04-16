import numpy as np

import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys, os


path = '/mnt/DataGuillaume/LMN-ADN/A5002/A5002-200304A/A5002-200304A.dat'


if os.path.exists(path):
	print(path)

frequency = 20000
n_channels = 96
channel = [39, 61, 68, 87]

f = open(path, 'rb')
startoffile = f.seek(0, 0)
endoffile = f.seek(0, 2)
bytes_size = 2		
n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
duration = n_samples/frequency
f.close()

with open(path, 'rb') as f:
	data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]

timestep = np.arange(0, len(data))/frequency


data = pd.DataFrame(index = timestep, data = data)

data.columns = channel

data.to_hdf('/mnt/DataGuillaume/LMN-ADN/A5002/A5002-200304A/Analysis/lfp_lmn.h5')
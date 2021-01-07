

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

data_directory 	= '/mnt/DataGuillaume/LMN-ADN/A5011'
info 			= pd.read_csv(os.path.join(data_directory,'A5011.csv'), index_col = 0)


s = info.index[-2]

print(s)
path = os.path.join(data_directory, s)
episodes = info.filter(like='Trial').loc[s].dropna().values
episodes[episodes != 'sleep'] = 'wake'
events = list(np.where(episodes != 'sleep')[0].astype('str'))


lfp 		= loadLFP(os.path.join(data_directory,s,s+'.dat'), 96, 0, 20000)

analogin_files = np.sort([f for f in os.listdir(path) if 'analogin' in f])

n_samples = []

for f in analogin_files:
	f = open(path + '/' + f, 'rb')
	startoffile = f.seek(0, 0)
	endoffile = f.seek(0, 2)
	bytes_size = 2        
	n_samples.append(int((endoffile-startoffile)/2/bytes_size))
	f.close()

n_samples = np.array(n_samples)

f = analogin_files[0]
with open(path+'/'+f, 'rb') as f:
	data = np.fromfile(f, np.uint16)


# analogin_0 missing
n_samples = np.hstack([[len(lfp)-np.sum(n_samples)],n_samples])

durations = nts.IntervalSet(
	start = [0, n_samples[0]/20000, np.sum(n_samples[0:-1])/20000],
	end = [n_samples[0]/20000, np.sum(n_samples[0:-1])/20000,np.sum(n_samples)/20000],
	time_units = 's')

csv_path = path + '/Epoch_TS.csv'

pd.DataFrame(durations.as_units('s')).to_csv(csv_path, header=None, index = None,
	float_format='%.6f')

# else
durations = nts.IntervalSet(
	start = [0, n_samples[0]/20000, np.sum(n_samples[0:-1])/20000],
	end = [n_samples[0]/20000, np.sum(n_samples[0:-1])/20000,np.sum(n_samples)/20000],
	time_units = 's')

csv_path = path + '/Epoch_TS.csv'

pd.DataFrame(durations.as_units('s')).to_csv(csv_path, header=None, index = None,
	float_format='%.6f')
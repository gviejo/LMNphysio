import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys, os
from functions import *
from wrappers import *
import h5py
from pylab import *


def loadTTLPulse(file, n_channels = 2, fs = 20000, track = 0, mscope = 1):
    """
        load ttl from analogin.dat
    """
    f = open(file, 'rb')
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2        
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    f.close()
    with open(file, 'rb') as f:
        data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
    
    ch_track = data[:,track].astype(np.int32)
    peaks,_ = scipy.signal.find_peaks(np.diff(ch_track), height=30000)
    timestep = np.arange(0, len(data))/fs
    peaks+=1
    ttl_track = pd.Series(index = timestep[peaks], data = data[peaks,track])    

    ch_mscope = data[:,mscope].astype(np.int32)
    peaks,_ = scipy.signal.find_peaks(np.abs(np.diff(ch_mscope)), height=30000)
    peaks+=1
    ttl_mscope = pd.Series(index = timestep[peaks], data = data[peaks,mscope])

    return ttl_track, ttl_mscope

def loadPosition(path, ttl, names = ['ry', 'rx', 'rz', 'x', 'y', 'z']):
	files = os.listdir(path)    
	csv_file = os.path.join(path, "".join(s for s in files if '.csv' in s))
	position = pd.read_csv(csv_file, header = [4,5], index_col = 1)
	if 1 in position.columns:
		position = position.drop(labels = 1, axis = 1)
	position = position[~position.index.duplicated(keep='first')]
	position.columns = names
	length = np.minimum(len(ttl), len(position))
	position = position.iloc[0:length]
	ttl = ttl.iloc[0:length]
	position.index = pd.Index(ttl.index[0:length])
	position[['ry', 'rx', 'rz']] *= (np.pi/180)
	position[['ry', 'rx', 'rz']] += 2*np.pi
	position[['ry', 'rx', 'rz']] %= 2*np.pi
	return position


path = '/home/guillaume/miniscoPy/A0624/12_3_2019'

##########################################################################################
# LOAD TTL PULSES
##########################################################################################
ttl_track, ttl_mscope = loadTTLPulse(os.path.join(path, 'analogin.dat'), 2)

##########################################################################################
# LOAD CALCIUM DATA
##########################################################################################
ms = {}
f = h5py.File(os.path.join(path, 'ms.mat'))['ms']
for k, v in f.items():
    ms[k] = np.array(v)

C = ms['FiltTraces'].T
C = pd.DataFrame(index = ttl_mscope.index, data = C[0:len(ttl_mscope)])
C =  C.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=100.0)	

C = pd.DataFrame(index = C.index.values[0:-1], data = np.diff(C, axis = 0)/C.values[0:-1])

##########################################################################################
# LOAD POSITION
##########################################################################################
position = loadPosition(path, ttl_track)

##########################################################################################
# CALCIUM TUNING CURVES
##########################################################################################
angle = position['ry'].loc[C.index[0]:C.index[-1]]
angle2 = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
angle2 = angle2.groupby(np.digitize(angle2.index.values, C.index.values)-1).mean()
angle2.index = pd.Index(C.index.values[0:-1])
angle2 = angle2.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)	
angle2	= angle2%(2*np.pi)

bins = np.linspace(0, 2*np.pi, 120)
tcurves = C.iloc[0:-1].groupby(np.digitize(angle2, bins)-1).sum()
tcurves.index = pd.Index(bins[0:-1]+np.diff(bins))
occupancy,_ = np.histogram(angle2.values, bins)
tcurves = tcurves/np.vstack(occupancy)

tcurves = smoothAngularTuningCurves(tcurves, window = 20, deviation = 3.0)
##########################################################################################
# PLOT
##########################################################################################
figure()
for i in range(tcurves.shape[1]):
	subplot(7,10,i+1)#, projection = 'polar')
	plot(tcurves[i])

import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys, os
sys.path.append('../')
from functions import *
from wrappers import *
import h5py
from pylab import *
import neuroseries as nts

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


path = '/home/guillaume/miniscoPy/A0624/12_3_2019_good'
# path = '/mnt/DataGuillaume/MINISCOPE/A0624/12_3_2019/H16_M35_S46'

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

C = ms['RawTraces'].T
C = pd.DataFrame(index = ttl_mscope.index, data = C[0:len(ttl_mscope)])
C =  C.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=100.0)   
Co = C.copy()
C = pd.DataFrame(index = C.index.values[0:-1], data = np.diff(C, axis = 0)/C.values[0:-1])

A = ms['SFPs']

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
angle2  = angle2%(2*np.pi)

bins = np.linspace(0, 2*np.pi, 60)
tcurves = []
for i in C:
    tmp = C[i].loc[angle2.index].groupby(np.digitize(angle2.values, bins)-1).mean()
    tmp.index = pd.Index(bins[0:-1]+np.diff(bins))
    occupancy,_ = np.histogram(angle2.values, bins)
    occupancy = occupancy/120
    tcurves.append(tmp)

tcurves = pd.concat(tcurves, 1)

angle2 = nts.Tsd(t = angle2.index.values, d = angle2.values, time_units = 's')

# FIND PEAKS
figure()
for i in C:
    # pk = scipy.signal.find_peaks(C[i], C[i].min() + (C[i].max() - C[i].min())/4)[0]
    pk = scipy.signal.find_peaks(C[i], C[i].max()/4)[0]
    pk = nts.Ts(t = C[i].index.values[pk], time_units = 's')
    # subplot(6,6,i+1)
    figure()
    subplot(311)
    plot(Co[i])
    subplot(312)
    plot(C[i])
    plot(C[i].loc[pk.as_units('s').index], 'o')
    subplot(313)
    plot(angle2)
    plot(angle2.realign(pk), 'o')


# tcurves = smoothAngularTuningCurves(tcurves, window = 20, deviation = 3.0)
##########################################################################################
# PLOT
##########################################################################################
figure()
for i in range(tcurves.shape[1]):
    subplot(6,6,i+1, projection = 'polar')
    plot(tcurves[i])


# figure(figsize = (15,5))
# subplot(121)
# imshow(A.sum(0).T)
# title("Spatial footprints")
# subplot(122)
# for i in C:
#     plot(C[i]+i)
# title("Calcium activity")
# ylabel("Neurons")
# xlabel("Time (s)")
# savefig("A0624_12_3_2019.pdf", dpi = 300)



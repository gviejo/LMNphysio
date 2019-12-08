import numpy as np
import pandas as pd
import sys, os
from scipy import signal
# sys.path.append('../')
# from functions import *
# from wrappers import *
import h5py
from pylab import *
import av


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
    peaks,_ = signal.find_peaks(np.diff(ch_track), height=30000)
    timestep = np.arange(0, len(data))/fs
    peaks+=1
    ttl_track = pd.Series(index = timestep[peaks], data = data[peaks,track])    

    ch_mscope = data[:,mscope].astype(np.int32)
    peaks,_ = signal.find_peaks(np.abs(np.diff(ch_mscope)), height=30000)
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

def get_video_array(video_file):
    video = av.open(video_file)
    stream = next(s for s in video.streams if s.type == 'video') 
    dims = (stream.duration, stream.format.height, stream.format.width)
    data = np.zeros(dims, dtype = np.float32)
    for i, packet in enumerate(video.demux(stream)):
        frame = packet.decode()[0].to_ndarray(format = 'bgr24')[:,:,0].astype(np.float32)
        data[i] = frame.reshape(dims[1], dims[2])
        if i+1 == dims[0]: break
    return data
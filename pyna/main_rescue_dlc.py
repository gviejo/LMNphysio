# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-08-10 17:16:25
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-08-17 16:24:26
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
from functions import *
import sys
from itertools import combinations, product
from matplotlib.pyplot import *

def load_ttl_pulse(    
    ttl_file,
    tracking_frequency,
    n_channels=1,
    channel=0,
    bytes_size=2,
    fs=20000.0,
    threshold=0.3,
):
    """
    Load TTLs from a binary file. Each TTLs is then used to reaassign the time index of tracking frames.

    Parameters
    ----------
    ttl_file : str
        File name
    n_channels : int, optional
        The number of channels in the binary file.
    channel : int, optional
        Which channel contains the TTL
    bytes_size : int, optional
        Bytes size of the binary file.
    fs : float, optional
        Sampling frequency of the binary file

    Returns
    -------
    pd.Series
        A series containing the time index of the TTL.
    """
    f = open(ttl_file, "rb")
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
    f.close()
    with open(ttl_file, "rb") as f:
        data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
    if n_channels == 1:
        data = data.flatten().astype(np.int32)
    else:
        data = data[:, channel].flatten().astype(np.int32)
    data = data / data.max()
    peaks, _ = scipy.signal.find_peaks(
        np.diff(data), height=threshold, distance=int(fs / (tracking_frequency * 2))
    )
    timestep = np.arange(0, len(data)) / fs
    peaks += 1
    ttl = pd.Series(index=timestep[peaks], data=data[peaks])
    return ttl



# path = '/mnt/DataRAID2/LMN-ADN/A5043/A5043-230301A'
path = '/mnt/ceph/users/gviejo/LMN-ADN/A5043/A5043-230315A'
        

data = nap.load_session(path, 'neurosuite')

spikes = data.spikes

wake_ep = data.epochs['wake']



# CSV FILE
csv_file = os.path.join(path, "A5043-230315A_1DLC_resnet50_A5043-230315AAug16shuffle1_100000.csv")

position = pd.read_csv(csv_file, header=[1, 2], index_col=0)
position = position[~position.index.duplicated(keep="first")]
position.columns = list(map(lambda x: "_".join(x), position.columns.values))

xy = np.vstack((position["objectA_x"] - position["bodypart3_x"], position["objectA_y"] - position["bodypart3_y"])).T

angle = np.arctan2(xy[:,0], xy[:,1])
angle = angle+np.pi

# # TTL
# ttl = load_ttl_pulse(os.path.join(path, "A5043-230315A_auxiliary.dat"), 120)

t = wake_ep.loc[0,'start']+np.arange(0, len(angle))/120

angle = nap.Tsd(t=t, d=angle)


ep = nap.IntervalSet(start = wake_ep.loc[0, 'start'], end = wake_ep.loc[0, 'start']+600)


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi), ep = ep)
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0)

figure()

for i in range(len(spikes)):
    subplot(6,6,i+1, projection='polar')
    plot(tuning_curves[i])

show()
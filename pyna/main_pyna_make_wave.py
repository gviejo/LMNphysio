# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-05-03 21:24:34
# @Last Modified by:   gviejo
# @Last Modified time: 2022-05-03 21:52:57
import scipy.io
import sys, os
import numpy as np
import pandas as pd
import pynapple as nap
from functions import *
import sys
from itertools import combinations, product
import wave
import struct

path = '/media/guillaume/LaCie/LMN-ADN/A5002/A5002-200303B'
data = nap.load_session(path, 'neurosuite')

filepath = "/media/guillaume/LaCie/LMN-ADN/A5002/A5002-200303B/A5002-200303B.dat"
#lfp = nap.load_eeg(filepath, 39, 96, 20000)
# lfp = data.load_lfp('A5002-200304A.dat', 39)

f = open(filepath, 'rb')
startoffile = f.seek(0, 0)
endoffile = f.seek(0, 2)
bytes_size = 2      
n_samples = int((endoffile-startoffile)/96/bytes_size)
duration = n_samples/20000.0
f.close()
timestep = np.arange(0, n_samples)/20000
timestep = nap.Tsd(t = timestep, d = np.arange(len(timestep)))

sws_ep = data.read_neuroscope_intervals('sws')

timestep = timestep.restrict(sws_ep)

fp = np.memmap(filepath, np.int16, 'r', shape = (n_samples, 96))        

sys.exit()

wav = wave.open("sample.wav", "w")
wav.setnchannels(1)
wav.setsampwidth(2)
wav.setframerate(sample_rate)

for i in range(200000):
	value = np.random.randint(-23767, 23757)
	data = struct.pack('<h', value)
	wav.writeframesraw(data)

wav.close()

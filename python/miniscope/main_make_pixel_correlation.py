import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys, os
# sys.path.append('../')
from msfunctions import *
# from wrappers import *
import h5py
from pylab import *
import av

# path = '/home/guillaume/miniscoPy/A0624/12_3_2019'
# path = '/mnt/DataGuillaume/MINISCOPE/A0624/12_3_2019/H16_M35_S46/A0624'
path = '/mnt/DataGuillaume/MINISCOPE/A0624/12_3_2019/H16_M35_S46'
video_file = '/mnt/DataGuillaume/MINISCOPE/A0624/12_3_2019/msvideo.avi'

##########################################################################################
# LOAD TTL PULSES
##########################################################################################
ttl_track, ttl_mscope = loadTTLPulse(os.path.join(path, 'analogin.dat'), 2)

##########################################################################################
# LOAD VIDEO
##########################################################################################
video = get_video_array(video_file)

##########################################################################################
# LOAD POSITION
##########################################################################################
position = loadPosition(path, ttl_track)


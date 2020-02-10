import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys, os
sys.path.append('../')
import hsluv
from msfunctions import *
# from wrappers import *
import h5py
from pylab import *
import av
from scipy.ndimage import gaussian_filter1d

path = '/home/guillaume/miniscoPy/A0624/12_3_2019'
video_file = '/home/guillaume/miniscoPy/A0624/12_3_2019/msvideo.avi'
# path = '/mnt/DataGuillaume/MINISCOPE/A0624/12_3_2019/H16_M35_S46/A0624'
# path = '/mnt/DataGuillaume/MINISCOPE/A0624/12_3_2019/H16_M35_S46'
# video_file = '/mnt/DataGuillaume/MINISCOPE/A0624/12_3_2019/msvideo.avi'

##########################################################################################
# LOAD TTL PULSES
##########################################################################################
ttl_track, ttl_mscope = loadTTLPulse(os.path.join(path, 'analogin.dat'), 2)

##########################################################################################
# LOAD VIDEO
##########################################################################################
video = get_video_array(video_file)

video = video[0:len(ttl_mscope)]

##########################################################################################
# LOAD POSITION
##########################################################################################
position = loadPosition(path, ttl_track)
angle = position['ry'].loc[ttl_mscope.index[0]:ttl_mscope.index[-1]]
angle2 = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))

angle2 = angle2.groupby(np.digitize(angle2.index.values, ttl_mscope.index.values)-1).mean()
angle2.index = pd.Index(ttl_mscope.index.values[0:-1])
angle2 = angle2.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)	
angle2	= angle2%(2*np.pi)

##########################################################################################
#
##########################################################################################

dims = (video.shape[1],video.shape[2])
T = video.shape[0]

bins = np.linspace(0, 2*np.pi, 60)
idx = pd.Index(bins[0:-1]+np.diff(bins))
occupancy = pd.Series(index = idx, data = np.histogram(angle2.values, bins)[0])


thetamap = np.zeros(dims)
rmap = np.zeros(dims)




for i in range(dims[0]):
	tmp = pd.DataFrame(index = ttl_mscope.index, data = video[:,i])
	baseline = tmp.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=2000.0)
	tmp2 = (tmp - baseline)/baseline
	tmp3 = tmp2.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)

	for j in range(dims[1]):
		print(i/dims[0], j/dims[1])
		tcurves = tmp3.iloc[0:-1,j].groupby(np.digitize(angle2, bins)-1).sum()
		tcurves.index = pd.Index(bins[0:-1]+np.diff(bins))
		tcurves = tcurves/occupancy

		tcurves -= tcurves.min()
		tcurves /= tcurves.max()	
		tcurves /= tcurves.sum()

		c = np.sum(tcurves.values*np.cos(tcurves.index.values))
		s = np.sum(tcurves.values*np.sin(tcurves.index.values))

		theta = np.arctan2(s, c)
		r = np.sqrt(c**2.0 + s**2)

		thetamap[i,j] = theta
		rmap[i,j] = r


from matplotlib.colors import hsv_to_rgb


	
H = np.copy(thetamap)
H[H<0] += 2*np.pi
# H = H*360/(2*np.pi)
H = H / (2*np.pi)
S = np.copy(rmap)
S -= S.min()
S /= S.max()
S = 1/(1+np.exp(-(S-0.5)*5))
V = np.ones_like(H)
HSV = np.dstack((H, S, V))
RGB = hsv_to_rgb(HSV)

# RGB = np.zeros_like(HSV)
# for i in range(H.shape[0]):
# 	for j in range(H.shape[1]):
# 		RGB[i,j] = hsluv.hsluv_to_rgb(HSV[i,j])

imshow(RGB, interpolation = 'nearest')
# sys.exit()


# for i in range(dims[0]):
# 	for j in range(dims[1]):
		
# 		tmp = pd.Series(index = ttl_mscope.index, data = video[:,i,j])
# 		baseline = tmp.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=2000.0)
# 		tmp2 = (tmp - baseline)/baseline
# 		tmp3 = tmp2.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)

# 		tcurves = tmp3.iloc[0:-1].groupby(np.digitize(angle2, bins)-1).sum()
# 		tcurves.index = pd.Index(bins[0:-1]+np.diff(bins))
# 		tcurves = tcurves/occupancy

# 		tcurves -= tcurves.min()
# 		tcurves /= tcurves.max()
# 		tcurves /= tcurves.sum()

# 		c = np.sum(tcurves.values*np.cos(tcurves.index.values))
# 		s = np.sum(tcurves.values*np.sin(tcurves.index.values))

# 		theta = np.arctan2(s, c)
# 		r = np.sqrt(c**2.0 + s**2)

# 		thetamap[i,j] = theta
# 		rmap[i,j] = r


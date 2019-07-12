import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

data_directory = '../data/A1400/A1407/'
info = pd.read_csv(data_directory+'A1407.csv')
info = info.set_index('Session')

path 								= '../data/A1400/A1407/A1407-190416'
spikes, shank 						= loadSpikeData(path)
n_channels, fs, shank_to_channel 	= loadXML(path)
episodes 							= info.filter(like='Trial').loc[path.split("/")[-1]].dropna().values
events								= list(np.where(episodes == 'wake')[0].astype('str'))
position 							= loadPosition(path, events, episodes)
wake_ep 							= loadEpoch(path, 'wake', episodes)
sleep_ep 							= loadEpoch(path, 'sleep')					



tuning_curves, velocity, edges 		= computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 61)
tokeep, stat 						= findHDCells(tuning_curves[1])

# tokeep 								= np.delete(tokeep, [7,8])

tcurves 							= tuning_curves[1][tokeep]
tcurves 							= smoothAngularTuningCurves(tcurves, 10, 2)
tcurves 							= tcurves[tcurves.columns[tcurves.idxmax().argsort().values]]

occupancy 							= np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]

decodedwake, proba_angle_wake		= decodeHD(tcurves, spikes, wake_ep, bin_size = 200, px = occupancy)

postsleep							= nts.IntervalSet(start = sleep_ep.loc[1,'start'], end = sleep_ep.loc[1,'end'])

decodedsleep, proba_angle_sleep 	= decodeHD(tcurves, spikes, postsleep, bin_size = 10, px = occupancy)

acceleration						= loadAuxiliary(path)
newsleep 							= refineSleepFromAccel(acceleration, sleep_ep)
newpostsleep						= postsleep.intersect(newsleep)


decodedsleep 						= decodedsleep.restrict(newpostsleep)  

entropy 							= (proba_angle_sleep*np.log2(proba_angle_sleep)).sum(1) + np.log2(proba_angle_sleep.shape[1])
filterd 							= entropy.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=100.0)
entropy 							= nts.Tsd(t = entropy.index.values*1000, d = entropy.values)
filterd								= nts.Tsd(t = filterd.index.values*1000, d = filterd.values)


ep1									= nts.IntervalSet(start=[8.84882e+9],end=[9e+9])
ep2 								= nts.IntervalSet(start=[9.91530e+9],end=[1.0114e+10])


lfp 								= pd.read_hdf(path+'/A1407-190416.h5') 
power 								= np.zeros(lfp.shape)
for n in range(n_channels):
	power[:,n] 						= np.abs(butter_bandpass_filter(lfp.values[:,n], 5, 15, 1250, 2))
power 								= pd.DataFrame(index = lfp.index, data = power)




lfpfilt 							= pd.DataFrame(index = lfp.index, data = )
power 								= nts.Tsd(lfpfilt.index.values, np.abs(lfpfilt.values))


sys.exit()

figure()
for i,n in zip(tcurves,np.arange(tcurves.shape[1])):
	subplot(3,4,n+1, projection = 'polar')
	plot(tcurves[i])
	
show()


figure()
ax = subplot(211)
plot(entropy)
plot(filterd)
axvline(ep1.iloc[0,0])
axvline(ep1.iloc[0,1])
axvline(ep2.iloc[0,0])
axvline(ep2.iloc[0,1])
subplot(212, sharex = ax)
plot(acceleration.iloc[:,0])
axvline(ep1.iloc[0,0])
axvline(ep1.iloc[0,1])
axvline(ep2.iloc[0,0])
axvline(ep2.iloc[0,1])

show()


figure()
ax = subplot(211)
for i,n in enumerate(tcurves.columns):
	pk = tcurves[n].idxmax()
	plot(spikes[n].restrict(ep2).fillna(pk).as_units('ms'), '|')
# ax2 = ax.twinx()
# plot(decodedsleep.restrict(ep1).as_units('ms'), 'o')

subplot(212, sharex = ax)
plot(filterd.restrict(ep2).as_units('ms'))


show()




# plot(spikes[7].restrict(ep1).fillna(7).as_units('ms'), '|')
# plot(spikes[8].restrict(ep1).fillna(8).as_units('ms'), '|')
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
from umap import UMAP
import sys
from matplotlib.colors import hsv_to_rgb
import hsluv
from sklearn.decomposition import PCA
from sklearn.manifold.isomap import Isomap


path = '/mnt/DataGuillaume/LMN-ADN/A5002/A5002-200304A'

episodes = ['sleep', 'wake', 'sleep']
events = [1]

spikes, shank 						= loadSpikeData(path)
n_channels, fs, shank_to_channel 	= loadXML(path)
position 							= loadPosition(path, events, episodes)
wake_ep 							= loadEpoch(path, 'wake', episodes)
sleep_ep 							= loadEpoch(path, 'sleep')					


# downsampleDatFile(path, n_channels = n_channels)


# sys.exit()

lfp 			= loadLFP(os.path.join(path,path.split("/")[-1]+'.eeg'), n_channels, 80, 1250, 'int16')
lfp 			= downsample(lfp, 1, 5)
lfp 			= lfp.restrict(wake_ep)
lfp_filt		= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 5, 15, 1250/5, 2))	
phase 			= getPhase(lfp, 6, 14, 16, 1250/5.)	

power	 		= nts.Tsd(lfp_filt.index.values, np.abs(lfp_filt.values))
enveloppe,dummy	= getPeaksandTroughs(power, 5)
index 			= (enveloppe.values > np.percentile(enveloppe, 30))*1.0
start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
tmp 			= np.where(end_cand != start_cand)
start_cand 		= enveloppe.index.values[start_cand[tmp]]
end_cand	 	= enveloppe.index.values[end_cand[tmp]]
good_ep			= nts.IntervalSet(start_cand, end_cand)
good_ep			= good_ep.drop_short_intervals(300000)

theta_wake_ep 	= wake_ep.intersect(good_ep).merge_close_intervals(30000).drop_short_intervals(1000000)


# wake_ep = nts.IntervalSet(start = wake_ep.loc[0, 'start'], end = wake_ep.loc[0, 'start']+wake_ep.tot_length()/3)

tokeep = list(np.where(shank >= 5)[0])

spikes_phase	= {n:phase.realign(spikes[n].restrict(theta_wake_ep), align = 'closest') for n in tokeep}

bins = np.linspace(-np.pi, np.pi, 51)
spikes_hist = {n:np.histogram(spikes_phase[n], bins)[0] for n in tokeep}
spikes_hist = pd.DataFrame.from_dict(spikes_hist)
spikes_hist.index = pd.Index(bins[0:-1] + np.diff(bins)/2)

# figure()
# for i,n in enumerate(tokeep):
# 	subplot(5,6,i+1)
# 	plot(spikes_hist[n], label = str(shank[n]) + " " + str(n))
# 	legend()

tokeep = [36, 42, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62]

####################################################################################################################
# BIN WAKE
####################################################################################################################
bin_size = 30
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)

neurons = tokeep
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

rate = np.sqrt(spike_counts/(bin_size*1e-3))
rate2 = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=5)
tmp = nts.TsdFrame(t = rate2.index.values, d = rate2.values, time_units = 'ms').restrict(theta_wake_ep).values
tmp = tmp.astype(np.float32)
# tmp = tmp - tmp.mean(0)
# tmp = tmp / tmp.std(0)

phase2 = phase.groupby(np.digitize(phase.as_units('ms').index.values, bins)).mean().values
phase2 = (phase2 + 2*np.pi)%(2*np.pi)
phase2 = nts.Tsd(t = bins[0:-1] + np.diff(bins)/2., d = phase2, time_units = 'ms').restrict(theta_wake_ep)

H = phase2.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

# RGB = np.array([hsluv.hsluv_to_rgb(HSV[i]) for i in range(len(HSV))])


# tmp = tmp[0:10000]

# pca = PCA(n_components = 5).fit_transform(tmp)
imap = Isomap(n_neighbors = 100, n_components = 2, n_jobs = -1).fit_transform(tmp)

scatter(imap[:,0], imap[:,1], c = RGB, marker = '.')

show()

sys.exit()

ump = UMAP(n_neighbors = 100).fit_transform(tmp)


figure()
scatter(ump[:,0], ump[:,1], s = 1)

show()


datatosave = {'ump':ump,
				'wakangle':wakangle}

# import _pickle as cPickle
# cPickle.dump(datatosave, open('../figures/figures_poster_2019/fig_4_ring_lmn.pickle', 'wb'))

# from sklearn.manifold import Isomap
# imap = Isomap(n_neighbors = 100, n_components = 2).fit_transform(tmp)
# figure()
# scatter(imap[:,0], imap[:,1], c= RGB, marker = '.', alpha = 0.5, linewidth = 0, s = 100)






# figure()
# for i,n in enumerate(tcurves.columns):
# 	subplot(3,5,i+1, projection = 'polar')
# 	plot(tcurves[n])
 



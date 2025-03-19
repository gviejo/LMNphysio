# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-03-17 14:23:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-03-19 16:48:32
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from pylab import *
import sys, os
sys.path.append("..")
from functions import *
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import yaml
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
warnings.simplefilter('ignore', category=UserWarning)


# %%
############################################################################################### 
# GENERAL infos
###############################################################################################
if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'

datasets = yaml.safe_load(
    open(os.path.join(
        data_directory,
        "datasets_OPTO.yaml"), "r"))['opto']


SI_thr = {
    'adn':0.2,
    'lmn':0.1,
    'psb':1.0
    }


sessions = {
	"adn-ipsilateral": "B2800/B2810/B2810-240925C",
	"adn-bilateral":   "B2800/B2809/B2809-240904B",
	"lmn-ipsilateral": "A8000/A8066/A8066-240216B"
	}

sessions_exs = {
	"B2800/B2810/B2810-240925C": nap.IntervalSet(8269, 8379),
    "B2800/B2809/B2809-240904B": nap.IntervalSet(6255, 6358),
    "A8000/A8066/A8066-240216B": nap.IntervalSet(5130, 5232),
}


# fig = figure(figsize=(20, 8))
# gs = GridSpec(3,1)

# for i, grp in enumerate(sessions.keys()[[1]]):

grp = "adn-bilateral"
s = sessions[grp]
st = grp.split("-")[0]

path = os.path.join(data_directory, "OPTO", s)
basename = os.path.basename(path)
filepath = os.path.join(path, "kilosort4", basename + ".nwb")


nwb = nap.load_file(filepath)

spikes = nwb['units']
spikes = spikes.getby_threshold("rate", 1)

position = []
columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
for k in columns:
    position.append(nwb[k].values)
position = np.array(position)
position = np.transpose(position)
position = nap.TsdFrame(
    t=nwb['x'].t,
    d=position,
    columns=columns,
    time_support=nwb['position_time_support'])

epochs = nwb['epochs']
wake_ep = epochs[epochs.tags == "wake"]
opto_ep = nwb["opto"]
sws_ep = nwb['sws']
nwb.close()

spikes = spikes[spikes.location == st]
opto_ep = opto_ep.intersect(sws_ep)
stim_duration = 1.0
opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start'])>=stim_duration-0.001]

tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves)
tcurves = tuning_curves
SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
spikes.set_info(SI)
spikes.set_info(max_fr = tcurves.max())

spikes = spikes.getby_threshold("SI", SI_thr[st])
spikes = spikes.getby_threshold("rate", 1.0)
spikes = spikes.getby_threshold("max_fr", 3.0)

tokeep = spikes.index
tcurves = tcurves[tokeep]

# peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
peaks = tcurves.idxmax()
order = np.argsort(peaks.sort_values().index)
spikes.set_info(order=order, peaks=peaks)
# tuning_curves.columns = [basename+"_"+str(i) for i in tuning_curves.columns]

# figure()
# for i, n in enumerate(tcurves.columns):
#     subplot(6,4,i+1, projection='polar')
#     plot(tcurves[n], label=f"{spikes.group[n]} {np.round(spikes.SI[n], 2)}")
#     legend()

#     plot([peaks[n], peaks[n]], [0, tcurves[n].max()])


# subplot(gs[i,0])
# for n in spikes.keys():
#     cl = hsv_to_rgb([spikes.peaks[n]/(2*np.pi), 1, 1])
#     plot(spikes[n].restrict(sessions_exs[sessions[grp]]).fillna(spikes.order[n]), '|', color=cl)


# [axvspan(s, e, alpha=0.1) for s, e in opto_ep.intersect(sessions_exs[sessions[grp]]).values]
# ax2 = gca().twinx()
# ax2.plot(position['ry'].restrict(sessions_exs[sessions[grp]]), '.')

# title(grp + " " + len(spikes))


figure()
for n in spikes.keys():
    cl = hsv_to_rgb([spikes.peaks[n]/(2*np.pi), 1, 1])
    plot(spikes[n].restrict(sws_ep).fillna(spikes.order[n]), '|', color=cl, markersize=20, mew=5)
[axvspan(s, e, alpha=0.5) for s, e in opto_ep.values]
# ylim(0, 2*np.pi)
ax2 = gca().twinx()
ax2.set_ylim(0, 2*np.pi)



    ###################################################

# tight_layout()
# savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_examples_wake.png"))
        

sys.exit()




# manifold

pre_opto = nap.IntervalSet(opto_ep.start-1.0, opto_ep.end)

count = spikes.restrict(pre_opto).count(0.02).smooth(0.05)
X = StandardScaler().fit_transform(count)
# A = np.unwrap(position['ry']).bin_average(0.2, ep=position.time_support[1])%(2*np.pi)
# RGB = getRGB(position['ry'], position.time_support[1], 0.2)

imap = MDS(n_components=2, dissimilarity="euclidean", random_state=42).fit_transform(X)
# imap = Isomap(n_components=2, n_neighbors=10).fit_transform(X)
# imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(X)
imap = nap.TsdFrame(t=count.t, d=imap)


figure()
scatter(imap[:,0].d, imap[:,1].d, color='grey')
tmp = imap.restrict(opto_ep)
scatter(tmp[:,0].d, tmp[:,1].d, color='red')
show()


from mpl_toolkits.mplot3d import axes3d
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(imap[:,0], imap[:,1], imap[:,2], c='grey', alpha=0.5)
tmp = imap.restrict(opto_ep)
ax.scatter(tmp[:,0], tmp[:,1], tmp[:,2], c='red')



# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-03-17 14:23:16
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-06-02 10:41:21
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
from scipy.ndimage import gaussian_filter1d, gaussian_filter
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
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'

datasets = yaml.safe_load(
    open(os.path.join(
        data_directory,
        "datasets_OPTO.yaml"), "r"))['opto']

SI_thr = {
    'adn': 0.2,
    'lmn': 0.1,
    'psb': 1.0
}

sessions = {
    "adn-ipsilateral": "B3700/B3704/B3704-240608A",  # 14
    "adn-bilateral": "B2800/B2809/B2809-240907A",  # 17
    # "lmn-ipsilateral": "A8000/A8062/A8062-231126B", #17
    "lmn-ipsilateral": "A8000/A8066/A8066-240216B"  # 27
}


############################################################################################################
############################################################################################################
############################################################################################################

fig = figure()
gs = GridSpec(3, 3)

for i, gr in enumerate(['lmn-ipsilateral', 'adn-ipsilateral', 'adn-bilateral']):

    s = sessions[gr]
    st = gr.split("-")[0]

    path = os.path.join(data_directory, "OPTO", s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    nwb = nap.load_file(filepath)
    spikes = nwb['units']
    spikes = spikes.getby_threshold("rate", 1)

    position = []
    columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    for k in columns:
        if k == 'ry':
            ry = nwb[k].values[:]
            position.append((ry + np.pi) % (2 * np.pi))
        else:
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
    opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start']) >= stim_duration - 0.001]

    sws2_ep = nap.IntervalSet(start=opto_ep.start - stim_duration, end=opto_ep.start)
    sws2_ep = sws2_ep.intersect(sws_ep)
    sws2_ep = sws2_ep.drop_short_intervals(stim_duration - 0.001)

    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
                                                 ep=position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves)
    tcurves = tuning_curves
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2 * np.pi))
    spikes.set_info(SI)
    spikes.set_info(max_fr=tcurves.max())

    spikes = spikes.getby_threshold("SI", SI_thr[st])
    spikes = spikes.getby_threshold("rate", 1.0)
    spikes = spikes.getby_threshold("max_fr", 3.0)

    tokeep = spikes.index
    tcurves = tcurves[tokeep]

    # peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
    peaks = tcurves.idxmax()
    order = np.argsort(peaks.reset_index(drop='True').sort_values().index)
    spikes.set_info(order=order, peaks=peaks)
    # tuning_curves.columns = [basename+"_"+str(i) for i in tuning_curves.columns]

    spikes = spikes[tokeep]


    # --------------------------
    # Binning, smoothing, normalization
    # --------------------------
    X_ = {}
    X_exs = {}
    models = {}
    idxs = {}

    special_bin_sizes = {
        "lmn-ipsilateral": 0.03,
        "adn-ipsilateral": 0.01,
        "adn-bilateral": 0.01
    }

    bin_sizes = {
        "wak": 0.3,
        "rem": 0.1,
        "sws": special_bin_sizes[gr],
    }

    for name, epochs in zip(
            [
                'wak',
                'sws',
            ],
            [
                position.time_support.loc[[0]],
                sws_ep,
            ]):

        X = np.sqrt(spikes.count(bin_sizes[name], epochs))
        X = X.smooth(bin_sizes[name]*3, norm=False)
        X = X - X.mean(0)
        X = X / X.std(0)
        X_[name] = X

    thr = np.percentile(X_['sws'].mean(1), 70)
    idx = (X_['sws'].mean(1) > thr).values
    X_['sws'] = X_['sws'][idx]

    # --------------------------
    # Dimensionality reduction
    # --------------------------
    proj = {}
    model = KernelPCA(n_components=2, kernel='cosine')
    proj['wak'] = model.fit_transform(X_['wak'])

    for name, epochs in zip(['opto', 'sws2'], [opto_ep, sws2_ep]):
        proj[name] = model.transform(X_['sws'].restrict(epochs).values)

    colors = {}

    colors['wak'] = getRGB(position['ry'], position.time_support.loc[[0]], bin_size=bin_sizes['wak'])

    tuning_curves2 = nap.compute_tuning_curves(
        spikes,
        position['ry'],
        bins=120,
        range=(0, 2 * np.pi)
    )
    decoded, P = nap.decode_bayes(
        tuning_curves=tuning_curves2,
        data=spikes.restrict(sws_ep),
        epochs=sws_ep,
        bin_size=bin_sizes['sws'],
        sliding_window_size=4,
    )



    decoded_color = getRGB(decoded, sws_ep, bin_size=bin_sizes['sws'])
    decoded_color = nap.TsdFrame(t=decoded.t, d=decoded_color)

    colors['sws2'] = decoded_color[idx].restrict(sws2_ep).values
    colors['opto'] = decoded_color[idx].restrict(opto_ep).values
    # --------------------------
    # PLOTTING
    # --------------------------
    for j, name in enumerate(['wak', 'sws2', 'opto']):
        print(gr, name)

        gs2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i, j], wspace=0.3, hspace=0.3)

        subplot(gs2[0, 0])
        scatter(
            proj[name][:, 0],
            proj[name][:, 1],
            c=colors[name],
            s=1,
            edgecolor='none',
            alpha=0.7
        )
        title(name)
        xticks([])
        yticks([])

        # Histogram
        subplot(gs2[0, 1])
        hist2d(proj[name][:, 0], proj[name][:, 1], bins=50, density=True, cmap='turbo')


tight_layout()
savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_opto_rings_sleep.pdf"))


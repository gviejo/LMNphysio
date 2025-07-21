# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-29 13:46:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-19 16:07:14
import numpy as np
import pandas as pd
import pynapple as nap
# import nwbmatic as ntm
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
warnings.simplefilter('ignore', category=UserWarning)





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
    'adn':0.2, 
    'lmn':0.1,
    'psb':1.0
    }


corr = {}

for st in ['adn', 'lmn']:
    
    corr[st] = {}

    for gr in ['opto', 'ctrl']:

        corr[st][gr] = {}

        for sd in ['ipsi', 'bilateral', 'contra']:

            try:
                dataset = datasets[st][gr][sd]['sleep']
            except:
                dataset = []

            if len(dataset):
                
                corr[st][gr][sd] = {'sws': [], 'opto': []}

                for s in dataset:
                    print(s)

                    ############################################################################################### 
                    # LOADING DATA
                    ###############################################################################################
                    path = os.path.join(data_directory, "OPTO", s)
                    basename = os.path.basename(path)
                    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
                    
                    if not os.path.exists(filepath):
                        print("missing ", st, gr, s)
                        sys.exit()
                    else:

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
                        # wake_ep = epochs[epochs.tags == "wake"]
                        opto_ep = nwb["opto"]
                        sws_ep = nwb['sws']
                        
                            
                    spikes = spikes[spikes.location == st]
                    
                    opto_ep = opto_ep.intersect(sws_ep)


                    ############################################################################################### 
                    # COMPUTING TUNING CURVES
                    ###############################################################################################
                    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
                    tuning_curves = smoothAngularTuningCurves(tuning_curves)
                    tcurves = tuning_curves
                    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
                    spikes.set_info(SI)
                    spikes.set_info(max_fr = tcurves.max())
                    

                    # figure()
                    # for i in range(len(spikes)):
                    #     subplot(6,6,i+1, projection='polar')
                    #     plot(tuning_curves[spikes.keys()[i]])
                    #     title(np.round(spikes.SI[spikes.keys()[i]], 3))
                    # show()

                    spikes = spikes.getby_threshold("SI", SI_thr[st])
                    spikes = spikes.getby_threshold("rate", 1.0)
                    spikes = spikes.getby_threshold("max_fr", 3.0)

                    tokeep = spikes.index
                    tcurves = tcurves[tokeep]
                    tuning_curves = tcurves
                    peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
                    order = np.argsort(peaks.values)
                    spikes.set_info(order=order, peaks=peaks)


                    stim_duration = 1.0
                    opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start'])>=stim_duration-0.001]


                    ############################################################################################### 
                    # AUTOCORR
                    ###############################################################################################        
                    if len(tokeep) > 1:
                        

                        sws2_ep = nap.IntervalSet(start=opto_ep.start-stim_duration, end=opto_ep.start)
                        sws2_ep = sws2_ep.intersect(sws_ep)
                        sws2_ep = sws2_ep.drop_short_intervals(stim_duration-0.001)

                        auc = {}
                        for e, iset, bin_size, window_size in zip(['sws', 'opto'], [sws2_ep, opto_ep], [0.002, 0.002], [0.06, 0.06]):

                            ac = nap.compute_autocorrelogram(spikes, binsize=bin_size, windowsize=window_size, ep=iset, norm=False)

                            ac.columns = pd.Index([basename+"_"+str(n) for n in ac.columns])

                            corr[st][gr][sd][e].append(ac)

                for e in ['sws', 'opto']:
                    corr[st][gr][sd][e] = pd.concat(corr[st][gr][sd][e], axis=1)



figure()
subplot(121)
for e in ['sws', 'opto']:
    tmp = corr['adn']['opto']['ipsi'][e]
    # tmp = tmp.apply(zscore, axis=0)
    plot(tmp.mean(1), label=e)

title("adn-ipsilateral")
legend()

subplot(122)
for e in ['sws', 'opto']:
    tmp = corr['adn']['opto']['bilateral'][e]
    # tmp = tmp.apply(zscore, axis=0)
    plot(tmp.mean(1), label=e)

title("adn-bilateral")
legend()

show()


# ##################################################################
# # FOR FIGURE 1
# ##################################################################
# datatosave = {
#     "allfr":allfr,
#     "change_fr":change_fr,
#     "allr":allr,
#     "corr":corr,
#     "baseline":baseline
# }


# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# file_name = "OPTO_SLEEP.pickle"

# import _pickle as cPickle

# with open(os.path.join(dropbox_path, file_name), "wb") as f:
#     cPickle.dump(datatosave, f)







#     
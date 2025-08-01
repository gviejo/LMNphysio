# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-29 13:46:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-08-01 14:34:47
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
import sys, os
sys.path.append("..")
from functions import *
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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

datasets = yaml.safe_load(open("/mnt/ceph/users/gviejo/datasets_OPTO.yaml", "r"))

datasets = datasets['opto']['psb']

SI_thr = {
    'adn':0.5, 
    'lmn':0.2,
    'psb':1.0
    }



alltc = {}
allmeta = {}
allfr = {}


for gr in ['opto', 'ctrl']:

    alltc[gr] = {}
    allmeta[gr] = {}
    allfr[gr] = {}

    for ep in ['wake', 'sleep']:

        alltc[gr][ep] = []
        allmeta[gr][ep] = []
        allfr[gr][ep] = []

        for s in datasets[gr]['ipsi'][ep]:
            print(gr, ep, s)    
            ############################################################################################### 
            # LOADING DATA
            ###############################################################################################            
            path = os.path.join(data_directory, "OPTO", s)
            basename = os.path.basename(path)
            filepath = os.path.join(path, "kilosort4", basename + ".nwb")
            
            if os.path.exists(filepath):            

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
                # sws_ep = nwb['sws']
                # rem_ep = nwb['rem']
                
                opto_ep = nwb['opto']

            else:            
                data = nap.load_session(path, 'neurosuite')
                spikes = data.spikes
                position = data.position
                wake_ep = data.epochs['wake'].loc[[0]]
                sleep_ep = data.epochs["sleep"]
                # sws_ep = data.read_neuroscope_intervals('sws')
                # rem_ep = data.read_neuroscope_intervals('rem')
                # down_ep = data.read_neuroscope_intervals('down')
                spikes = spikes.getby_threshold("rate", 1.0)
                idx = spikes.metadata[spikes.metadata["location"].str.contains("psb|PSB")].index.values
                spikes = spikes[idx]
            
                try:
                    opto_ep = nap.load_file(os.path.join(path, os.path.basename(path))+"_opto_ep.npz")
                except:
                    opto_ep = []
                    epoch = 0
                    while len(opto_ep) == 0:
                        try:
                            opto_ep = loadOptoEp(path, epoch=epoch, n_channels = 2, channel = 0)
                            opto_ep = opto_ep.intersect(data.epochs[ep])
                        except:
                            pass                    
                        epoch += 1
                        if epoch == 10:
                            print("No opto ep found")
                            sys.exit()

                    opto_ep.save(os.path.join(path, os.path.basename(path))+"_opto_ep")

            ############################################################################################### 
            # COMPUTING TUNING CURVES
            ###############################################################################################
            tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
            tuning_curves = smoothAngularTuningCurves(tuning_curves)    
            tcurves = tuning_curves        
            SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
            spikes.set_info(SI)
            spikes.set_info(max_fr = tcurves.max())


            tokeep = spikes.index
            tcurves = tcurves[tokeep]
            peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
            order = np.argsort(peaks.values)
            spikes.set_info(order=order, peaks=peaks)

            ############################################################################################### 
            # FIRING RATE MODULATION
            ###############################################################################################    
            # stim_duration = np.round(opto_ep.loc[0,'end'] - opto_ep.loc[0,'start'], 3)
            if ep == 'wake':
                stim_duration = 10.0
            else:
                stim_duration = 1.0

            # print(stim_duration)

            # peth = nap.compute_perievent(spikes[tokeep], nap.Ts(opto_ep.start), minmax=(-stim_duration, 2*stim_duration))
            # frates = pd.DataFrame({n:np.sum(peth[n].count(0.05), 1).values for n in peth.keys()})

            frates = nap.compute_eventcorrelogram(spikes[tokeep], nap.Ts(opto_ep.start), stim_duration/20., stim_duration*2, norm=True)
                    
            frates.columns = [basename+"_"+str(i) for i in frates.columns]

            #######################
            # SAVING
            #######################        
            allfr[gr][ep].append(frates)
            metadata = spikes.metadata
            metadata.index = frates.columns
            allmeta[gr][ep].append(metadata)
            tuning_curves.columns = frates.columns
            alltc[gr][ep].append(tuning_curves)        
        

        print(gr, ep)        

        allfr[gr][ep] = pd.concat(allfr[gr][ep], axis=1)
        allmeta[gr][ep] = pd.concat(allmeta[gr][ep], axis=0)
        alltc[gr][ep] = pd.concat(alltc[gr][ep], axis=1)




# print(scipy.stats.wilcoxon(corr.iloc[:,-2], corr.iloc[:,-1]))


for gr in ['opto', 'ctrl']:
    figure()
    gs = GridSpec(2, 2)
    for i, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(-4,14), slice(-1,2)], [slice(-4,0), slice(-1,0)]):
        order = allmeta[gr][ep].sort_values(by="SI").index.values
        tmp = allfr[gr][ep][order]#.loc[sl]
        tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1, mode='constant'))
        subplot(gs[0,i])
        plot(tmp)
        title(ep)
        subplot(gs[1,i])    
        # tmp = tmp - tmp.loc[msl].mean(0)
        # tmp = tmp / tmp.std(0)    
        imshow(tmp.values.T, cmap = 'jet', aspect='auto')
        title(ep)
    tight_layout()

    savefig(os.path.expanduser(f"~/Dropbox/LMNphysio/summary_opto/fig_PSB_{gr.upper()}.png"))

show()


##################################################################
# FOR FIGURE 1
##################################################################
datatosave = {
    "allfr":allfr,
    "allmeta":allmeta,
    "alltc":alltc
}


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
file_name = "OPTO_PSB.pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)
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
import nwbmatic as ntm
import yaml


def loadXML(path):
    """
    path should be the folder session containing the XML file
    Function returns :
        1. the number of channels
        2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
            eeg file first if both are present or both are absent
        3. the mappings shanks to channels as a dict
    Args:
        path : string

    Returns:
        int, int, dict
    """
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(path)
    xmlfiles = [f for f in listdir if f.endswith('.xml')]
    if not len(xmlfiles):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, xmlfiles[0])

    from xml.dom import minidom
    xmldoc = minidom.parse(new_path)
    nChannels = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
    fs_dat = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
    fs_eeg = xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[
        0].firstChild.data
    if os.path.splitext(xmlfiles[0])[0] + '.dat' in listdir:
        fs = fs_dat
    elif os.path.splitext(xmlfiles[0])[0] + '.eeg' in listdir:
        fs = fs_eeg
    else:
        fs = fs_eeg
    shank_to_channel = {}
    groups = xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[
        0].getElementsByTagName('group')
    for i in range(len(groups)):
        shank_to_channel[i] = np.sort(
            [int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
    return int(nChannels), int(fs), shank_to_channel


def loadMeanWaveforms(path):
    """
    load waveforms
    quick and dirty
    """
    import scipy.io
    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, 'Analysis/')

    # Creating /Analysis/ Folder here if not already present
    if not os.path.exists(new_path): os.makedirs(new_path)
    files = os.listdir(path)
    clu_files = np.sort([f for f in files if 'clu' in f and f[0] != '.'])
    spk_files = np.sort([f for f in files if 'spk' in f and f[0] != '.'])
    clu1 = np.sort([int(f.split(".")[-1]) for f in clu_files])
    clu2 = np.sort([int(f.split(".")[-1]) for f in spk_files])
    if len(clu_files) != len(spk_files) or not (clu1 == clu2).any():
        print("Not the same number of clu and res files in " + path + "; Exiting ...")
        sys.exit()

        # XML INFO
    n_channels, fs, shank_to_channel = loadXML(path)
    from xml.dom import minidom
    xmlfile = os.path.join(path, [f for f in files if f.endswith('.xml')][0])
    xmldoc = minidom.parse(xmlfile)
    nSamples = int(xmldoc.getElementsByTagName('nSamples')[0].firstChild.data)  # assuming constant nSamples

    import xml.etree.ElementTree as ET
    root = ET.parse(xmlfile).getroot()

    count = 0
    meanwavef = {}
    maxch = []
    for i, s in zip(range(len(clu_files)), clu1):
        clu = np.genfromtxt(os.path.join(path, clu_files[i]), dtype=np.int32)[1:]
        mwf = []
        mch = []
        if np.max(clu) > 1:
            # load waveforms
            file = os.path.join(path, spk_files[i])
            f = open(file, 'rb')
            startoffile = f.seek(0, 0)
            endoffile = f.seek(0, 2)
            bytes_size = 2
            n_samples = int((endoffile - startoffile) / bytes_size)
            f.close()
            n_channel = len(root.findall('spikeDetection/channelGroups/group')[s - 1].findall('channels')[0])

            data = np.memmap(file, np.int16, 'r', shape=(len(clu), nSamples, n_channel))

            # data = np.fromfile(open(file, 'rb'), np.int16)
            # data = data.reshape(len(clu),nSamples,n_channel)

            tmp = np.unique(clu).astype(int)
            idx_clu = tmp[tmp > 1]
            idx_col = np.arange(count, count + len(idx_clu))
            for j, k in zip(idx_clu, idx_col):
                print(i, j)
                # take only a subsample of spike if too big
                idx = np.sort(np.random.choice(np.where(clu == j)[0], 100))
                meanw = data[idx, :, :].mean(0)
                ch = np.argmax(np.max(np.abs(meanw), 0))
                mwf.append(meanw.flatten())
                mch.append(ch)
            mwf = pd.DataFrame(np.array(mwf).T)
            mwf.columns = pd.Index(idx_col)
            mch = pd.Series(index=idx_col, data=mch)
            count += len(idx_clu)
            meanwavef[i] = mwf
            maxch.append(mch)

    # meanwavef = pd.concat(meanwavef, 1)
    # maxch = pd.concat(maxch)
    # meanwavef.to_hdf(os.path.join(new_path, 'MeanWaveForms.h5'), key='waveforms', mode='w')
    # maxch.to_hdf(os.path.join(new_path, 'MaxWaveForms.h5'), key='channel', mode='w')
    return meanwavef


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
allmwf = {}

for gr in ['opto', 'ctrl']:

    alltc[gr] = {}
    allmeta[gr] = {}
    allfr[gr] = {}
    allmwf[gr] = {}

    for ep in ['wake', 'sleep']:

        alltc[gr][ep] = []
        allmeta[gr][ep] = []
        allfr[gr][ep] = []
        allmwf[gr][ep] = []

        for s in datasets[gr]['ipsi'][ep]:
            print(gr, ep, s)    
            ############################################################################################### 
            # LOADING DATA
            ###############################################################################################            
            path = os.path.join(data_directory, "OPTO", s)
            basename = os.path.basename(path)
            # filepath = os.path.join(path, "kilosort4", basename + ".nwb")
            
            # if os.path.exists(filepath):
            #
            #     nwb = nap.load_file(filepath)
            #
            #     spikes = nwb['units']
            #     spikes = spikes.getby_threshold("rate", 1)
            #
            #
            #     position = []
            #     columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
            #     for k in columns:
            #         position.append(nwb[k].values)
            #     position = np.array(position)
            #     position = np.transpose(position)
            #     position = nap.TsdFrame(
            #         t=nwb['x'].t,
            #         d=position,
            #         columns=columns,
            #         time_support=nwb['position_time_support'])
            #
            #     epochs = nwb['epochs']
            #     wake_ep = epochs[epochs.tags == "wake"]
            #     # sws_ep = nwb['sws']
            #     # rem_ep = nwb['rem']
            #
            #     opto_ep = nwb['opto']
            #
            #     meanwavef = loadMeanWaveforms(os.path.join(path, "kilosort4"))
            #     meanwavef = meanwavef[0][spikes.index]
            #
            # else:
            data = ntm.load_session(path, 'neurosuite')
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
            #
            # meanwavef = loadMeanWaveforms(path)
            # meanwavef = meanwavef[0][spikes.index]

            meanwavef, maxch = data.load_mean_waveforms()

            tmp = []
            for k in spikes.keys():
                tmp.append(meanwavef[k][maxch[k]])
            tmp = pd.concat(tmp, axis=1)
            meanwavef = tmp

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

            meanwavef.columns = frates.columns

            #######################
            # SAVING
            #######################        
            allfr[gr][ep].append(frates)
            metadata = spikes.metadata
            metadata.index = frates.columns
            allmeta[gr][ep].append(metadata)
            tuning_curves.columns = frates.columns
            alltc[gr][ep].append(tuning_curves)
            allmwf[gr][ep].append(meanwavef)
        

        print(gr, ep)        

        allfr[gr][ep] = pd.concat(allfr[gr][ep], axis=1)
        allmeta[gr][ep] = pd.concat(allmeta[gr][ep], axis=0)
        alltc[gr][ep] = pd.concat(alltc[gr][ep], axis=1)
        allmwf[gr][ep] = pd.concat(allmwf[gr][ep], axis=1)



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
    "alltc":alltc,
    "allmwf":allmwf
}


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
file_name = "OPTO_PSB.pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)
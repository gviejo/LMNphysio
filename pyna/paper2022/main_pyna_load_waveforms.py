# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-05-06 15:59:38
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-05-13 14:26:33
import numpy as np
import pandas as pd
import pynapple as nap
import os, sys



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
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(path)
    xmlfiles = [f for f in listdir if f.endswith('.xml')]
    if not len(xmlfiles):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, xmlfiles[0])
    
    from xml.dom import minidom 
    xmldoc      = minidom.parse(new_path)
    nChannels   = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
    fs_dat      = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
    fs_eeg      = xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data  
    if os.path.splitext(xmlfiles[0])[0] +'.dat' in listdir:
        fs = fs_dat
    elif os.path.splitext(xmlfiles[0])[0] +'.eeg' in listdir:
        fs = fs_eeg
    else:
        fs = fs_eeg
    shank_to_channel = {}
    groups      = xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
    for i in range(len(groups)):
        shank_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
    return int(nChannels), int(fs), shank_to_channel

def loadMeanWaveforms(path):
    """
    load waveforms
    quick and dirty 
    """
    import scipy.io
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()    
    new_path = os.path.join(path, 'Analysis/')

    # Creating /Analysis/ Folder here if not already present
    if not os.path.exists(new_path): os.makedirs(new_path)
    files = os.listdir(path)
    clu_files     = np.sort([f for f in files if 'clu' in f and f[0] != '.'])   
    spk_files     = np.sort([f for f in files if 'spk' in f and f[0] != '.'])
    clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
    clu2         = np.sort([int(f.split(".")[-1]) for f in spk_files])
    if len(clu_files) != len(spk_files) or not (clu1 == clu2).any():
        print("Not the same number of clu and res files in "+path+"; Exiting ...")
        sys.exit()  

    # XML INFO
    n_channels, fs, shank_to_channel    = loadXML(path)
    from xml.dom import minidom 
    xmlfile = os.path.join(path, [f for f in files if f.endswith('.xml')][0])
    xmldoc      = minidom.parse(xmlfile)
    nSamples    = int(xmldoc.getElementsByTagName('nSamples')[0].firstChild.data) # assuming constant nSamples

    import xml.etree.ElementTree as ET
    root = ET.parse(xmlfile).getroot()


    count = 0
    meanwavef = {}
    maxch = []
    for i, s in zip(range(len(clu_files)),clu1):
        clu = np.genfromtxt(os.path.join(path,clu_files[i]),dtype=np.int32)[1:]
        mwf = []
        mch = []        
        if np.max(clu)>1:
            # load waveforms
            file = os.path.join(path, spk_files[i])
            f = open(file, 'rb')
            startoffile = f.seek(0, 0)
            endoffile = f.seek(0, 2)
            bytes_size = 2
            n_samples = int((endoffile-startoffile)/bytes_size)
            f.close()           
            n_channel = len(root.findall('spikeDetection/channelGroups/group')[s-1].findall('channels')[0])

            data = np.memmap(file, np.int16, 'r', shape = (len(clu), nSamples, n_channel))

            #data = np.fromfile(open(file, 'rb'), np.int16)
            #data = data.reshape(len(clu),nSamples,n_channel)

            tmp = np.unique(clu).astype(int)
            idx_clu = tmp[tmp>1]
            idx_col = np.arange(count, count+len(idx_clu))          
            for j,k in zip(idx_clu, idx_col):
                print(i,j)
                # take only a subsample of spike if too big             
                idx = np.sort(np.random.choice(np.where(clu==j)[0], 100))
                meanw = data[idx,:,:].mean(0)
                ch = np.argmax(np.max(np.abs(meanw), 0))
                mwf.append(meanw.flatten())
                mch.append(ch)
            mwf = pd.DataFrame(np.array(mwf).T)
            mwf.columns = pd.Index(idx_col)
            mch = pd.Series(index = idx_col, data = mch)
            count += len(idx_clu)
            meanwavef[i] = mwf
            maxch.append(mch)

    # meanwavef = pd.concat(meanwavef, 1)
    # maxch = pd.concat(maxch)  
    # meanwavef.to_hdf(os.path.join(new_path, 'MeanWaveForms.h5'), key='waveforms', mode='w')
    # maxch.to_hdf(os.path.join(new_path, 'MaxWaveForms.h5'), key='channel', mode='w')
    return meanwavef



############################################################################################### 
# LOADING DATA
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.genfromtxt(os.path.join(data_directory,'datasets_UFO.list'), delimiter = '\n', dtype = str, comments = '#')


# 53 1 73

for s in datasets:
    path = os.path.join(data_directory, s)
    data = nap.load_session(path, 'neurosuite')

    meanwavef = loadMeanWaveforms(path)


    import _pickle as cPickle

    cPickle.dump(meanwavef, open(os.path.join(path, 'pynapplenwb', 'MeanWaveForms.pickle'), 'wb'))





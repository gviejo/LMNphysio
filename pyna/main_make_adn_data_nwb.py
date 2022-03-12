# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-02 18:27:24
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-03-03 11:06:23
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import sys, os
import pandas as pd
import wrappers_thalamus_physio as wtp
import scipy
import scipy.io
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.behavior import Position, SpatialSeries, CompassDirection
from pynwb.file import Subject
from pynwb.epoch import TimeIntervals
import datetime
from xml.dom import minidom

data_directory  = '/mnt/DataRAID/MergedData/'
datasets        = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

data_output = '/mnt/DataRAID/ADN'

allsessions = []
n_hd_neurons = []

for m in ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']:
    sessions        = [n.split("/")[1] for n in datasets if m in n]
    if not os.path.exists(os.path.join(data_output, m)):
        os.mkdir(os.path.join(data_output, m))

    for s in sessions:        
        print(s)        
        generalinfo         = scipy.io.loadmat(data_directory+m+"/"+s+'/Analysis/GeneralInfo.mat')      
        shankStructure      = wtp.loadShankStructure(generalinfo)
        spikes,shank        = wtp.loadSpikeData(data_directory+m+"/"+s+'/Analysis/SpikeData.mat', shankStructure['thalamus'])                       
        wake_ep             = wtp.loadEpoch(data_directory+m+'/'+s, 'wake')
        sleep_ep            = wtp.loadEpoch(data_directory+m+'/'+s, 'sleep')
        try:
            sws_ep              = wtp.loadEpoch(data_directory+m+'/'+s, 'sws')
        except:
            pass
        rem_ep              = wtp.loadEpoch(data_directory+m+'/'+s, 'rem')

        hd_info             = scipy.io.loadmat(data_directory+m+'/'+s+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
        hd_info_neuron      = np.array([hd_info[n] for n in spikes.keys()])     

        try:
            data        = pd.read_csv(data_directory+m+"/"+s+"/"+s+ ".csv", delimiter = ',', header = None, index_col = [0])            
            data.columns = ['ry']
        except:
            pass

        if np.sum(hd_info_neuron)>5:

            path1 = os.path.join(data_output, m, s)
            if not os.path.exists(path1):
                os.mkdir(path1)

            path = os.path.join(path1, 'pynapplenwb')
            if not os.path.exists(path):
                os.mkdir(path)

            nwbfilepath = os.path.join(path, s+'.nwb')
        
            if not os.path.exists(nwbfilepath):                
        
                nwbfile = NWBFile(
                    session_start_time = datetime.datetime.now(datetime.timezone.utc),
                    session_description=s,
                    identifier=s,
                )

                direction = CompassDirection()
                for c in ['ry']:
                    tmp = SpatialSeries(
                        name=c, 
                        data=data[c].values, 
                        timestamps=data.index.values, 
                        unit='radian',
                        reference_frame='')
                    direction.add_spatial_series(tmp)

                nwbfile.add_acquisition(direction)

                # Adding time support of position as TimeIntervals
                epochs = nap.IntervalSet(start = data.index[0], end = data.index[-1])
                position_time_support = TimeIntervals(
                    name="position_time_support",
                    description="The time support of the position i.e the real start and end of the tracking"
                    )
                for i in epochs.index:
                    position_time_support.add_interval(
                        start_time=epochs.loc[i,'start'],
                        stop_time=epochs.loc[i,'end'],
                        tags=str(i)
                        )

                nwbfile.add_time_intervals(position_time_support)

                epochs = {'wake':wake_ep, 'sleep':sleep_ep}

                # Epochs
                for ep in epochs.keys():
                    epoch = epochs[ep].as_units('s')
                    for i in epochs[ep].index:
                        nwbfile.add_epoch(
                            start_time=epoch.loc[i,'start'],
                            stop_time=epoch.loc[i,'end'],
                            tags=[ep] # This is stupid nwb who tries to parse the string
                            )


                for iset, name in zip([sws_ep, rem_ep], ['sws', 'rem']):
                    epochs = iset.as_units('s')
                    time_intervals = TimeIntervals(
                        name=name,
                        )
                    for i in epochs.index:
                        time_intervals.add_interval(
                            start_time=epochs.loc[i,'start'],
                            stop_time=epochs.loc[i,'end'],
                            tags=str(i)
                            )

                    nwbfile.add_time_intervals(time_intervals)

                # XML
                listdir = os.listdir(data_directory+m+'/'+s)
                xmlfiles = [f for f in listdir if f == s+'.xml']
                new_path = os.path.join(data_directory+m+'/'+s, xmlfiles[0])
                
                
                xmldoc      = minidom.parse(new_path)
                nChannels   = int(xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data)
                fs_dat      = int(xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data)
                fs_eeg      = int(xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data)

                group_to_channel = {}
                groups      = xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
                for i in range(len(groups)):
                    group_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])


                #spikes
                electrode_groups = {}

                location = {}
                for st in shankStructure:
                    for g in shankStructure[st]:
                        location[g] = st

                shank_with_hd = np.unique(shank[spikes.keys()].flatten()[hd_info_neuron == 1.0])
                for st in shank_with_hd:
                    location[st] = 'adn'

                for g in group_to_channel:
                    if g not in location.keys():
                        location[g] = ''
                    device = nwbfile.create_device(
                        name=s+'-'+str(g),
                        description='',
                        manufacturer=''
                        )

                    electrode_groups[g] = nwbfile.create_electrode_group(
                        name='group'+str(g),
                        description='',
                        position=None,
                        location=location[g],
                        device=device
                        )

                    for idx in group_to_channel[g]:
                        nwbfile.add_electrode(id=idx,
                                              x=0.0, y=0.0, z=0.0,
                                              imp=0.0,
                                              location='', 
                                              filtering='none',
                                              group=electrode_groups[g])

                
                # Adding units
                nwbfile.add_unit_column('location', 'the anatomical location of this unit')
                nwbfile.add_unit_column('group', 'the group of the unit')
                for u in spikes.keys():
                    nwbfile.add_unit(
                        id=u,
                        spike_times=spikes[u].as_units('s').index.values,                
                        electrode_group=electrode_groups[shank[u,0]],
                        location=location[shank[u,0]],                    
                        group=shank[u,0]
                        )            
                
                with NWBHDF5IO(nwbfilepath, 'w') as io:
                    io.write(nwbfile)

                # sys.exit()
                allsessions.append('ADN/'+m+'/'+s+' # '+str(int(np.sum(hd_info_neuron))))

with open('/mnt/DataGuillaume/datasets_ADN.list', 'a') as f:
    for line in allsessions:
        f.write(line+'\n')

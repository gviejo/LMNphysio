#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:25:17 2022

@author: dl2820
"""
#from LinearDecoder import linearDecoder
import numpy as np
import pynapple as nap

#Load the data (from NWB)
data_directory = '/Users/dl2820/Desktop/LinearDecoderTutorial/A2929-200711'
data = nap.load_session(data_directory, 'neurosuite')

spikes = data.spikes
position = data.position
epochs = data.epochs

#%%

#Convert spikes to rates
bin_dt = 0.2 #200ms bins
rates = spikes.count(bin_dt)

#New Tsd with HD signal at same timepoints as rates
HD = rates.value_from(position.ry)

#Restrict Rates to only times we have HD info
rates = rates.restrict(HD.time_support)
 
#%%
#Bin HD and put it into a Tsd
numHDbins = 25
HDbinedges = np.linspace(0,2*np.pi,numHDbins+1)

HD_binned = np.digitize(HD.values,HDbinedges)-1 #(-1 for 0-indexed category)
HD_binned = nap.Tsd(d=HD_binned, t=HD.index.values)

#Pynapple Question: why doesn't this work? Do we want it to?
#HD['binned'] = np.digitize(HD.values,HDbinedges)

#%%
#Separate Train and Test data
holdout = 0.1 #percentage of data to hold out for test set

train_rates = rates.head(np.int32((1-holdout)*len(rates)))
train_HD = HD_binned.head(np.int32((1-holdout)*len(rates)))

test_rates = rates.tail(np.int32((holdout)*len(rates)))
test_HD = HD_binned.tail(np.int32((holdout)*len(rates)))

#%%
import matplotlib.pyplot as plt
#Plot data
plt.figure()

plt.subplot(2,1,1)
#Pynapple Question: do we want this to pull timestamps, like plot?
plt.imshow(rates.T, aspect='auto')  
plt.ylabel('Cell')

plt.subplot(2,1,2)
plt.plot(train_HD)
plt.plot(test_HD)
plt.xlabel('t (s)')
plt.ylabel('HD (bin)')

plt.show()

#%%
from LinearDecoder import linearDecoder
#%%
N_units = len(spikes)
decoder = linearDecoder(N_units,numHDbins)

#Train the decoder 
batchSize = 0.75
numBatches = 20000
decoder.train(train_rates.values, train_HD.values, 
              batchSize=batchSize, numBatches=numBatches,
              Znorm=False)
#%%
#Decode HD from test set
decoded, p = decoder.decode(test_rates.values)

#Calculate decoding error
decode_error = np.abs(decoded - test_HD.values)

#%%


import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,2,4)
plt.hist(decode_error)
plt.xlabel('Error (bins)')
plt.subplot(2,1,1)
plt.imshow(p.T, aspect='auto',interpolation='none')
plt.plot(test_HD.values,'r')
plt.plot(decoded)
#plt.xlim(0,100)
#plt.xlim(1000,1500)
plt.show()

#%%

#%%
plt.figure()
plt.plot(decoder.model.weight.detach().numpy()[:,2])

#%% Position! 


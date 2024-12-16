# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-12-02 14:55:27
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-05 15:28:35

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from pylab import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.stats import zscore
import os

dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")

data = cPickle.load(open(os.path.join(dropbox_path, 'GLM_HMM_LMN_05-12-2024.pickle'), 'rb'))

allradn = data['allr']['adn']
allrlmn = data['allr']['lmn']


fz = {'adn':np.arctanh(allradn[['wak', 'sws', 'ep1', 'ep2']]), 'lmn':np.arctanh(allrlmn[['wak', 'sws', 'ep1', 'ep2']])}
# fz = {'adn':allradn[['wak', 'sws']], 'lmn':allrlmn[['wak', 'sws']]}
angs = {'adn':allradn['ang'], 'lmn':allrlmn['ang']}

zbins = np.linspace(-1.5, 1.5, 100)

angbins = np.linspace(0, np.pi, 4)



##############################################
# Hist of Fisher Z
##############################################
p = {}
meanp = {}

for i, g in enumerate(['adn', 'lmn']):

    groups = fz[g].groupby(np.digitize(angs[g], angbins))

    for j in range(angbins.shape[0]-1):

        idx = groups.groups[j+1]
        z = fz[g].loc[idx]
        
        for k in ['ep1', 'ep2']:

            count = np.histogram(np.abs(z['wak'] - z[k]), zbins)[0]
            count = count/np.sum(count)  

            p[g+'-'+k+'-'+str(j)] = count
            meanp[g+'-'+k+'-'+str(j)] = np.mean(z['wak'] - z[k])

p = pd.DataFrame.from_dict(p)
p = p.set_index(pd.Index(zbins[0:-1] + np.diff(zbins)/2))
p = p.rolling(10, win_type='gaussian').sum(std=1)


####
colors = rcParams['axes.prop_cycle'].by_key()['color']

figure()
gs = GridSpec(3,2)

for i in range(angbins.shape[0]-1):

    for j, g in enumerate(['adn', 'lmn']):

        subplot(gs[i,j])

        for k, ls in zip(['ep1', 'ep2'], ['-', '--']):
            
            step(p.index.values, p[g+'-'+k+'-'+str(i)], 
                #np.mean(np.diff(zbins)), 
                label=k, alpha = 0.5, color=colors[j], linestyle=ls
                )
        
        legend()
        if i == 0: 
            title(g)
        # axvline(meanp[g+"-"+str(i)], linewidth=1, color=colors[j])
    
        

show()




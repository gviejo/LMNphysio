# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-12-02 14:55:27
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-03 16:00:20

import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.stats import zscore

dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
allrlmn = data['allr']

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'rb'))
allradn = data['allr']

data = cPickle.load(open(os.path.join(dropbox_path, 'All_CC_LMN.pickle'), 'rb'))
allcc_lmn = data['allcc']

data = cPickle.load(open(os.path.join(dropbox_path, 'All_CC_ADN.pickle'), 'rb'))
allcc_adn = data['allcc']

allcc = {'adn':allcc_adn, 'lmn':allcc_lmn}

fz = {'adn':np.arctanh(allradn[['wak', 'sws']]), 'lmn':np.arctanh(allrlmn[['wak', 'sws']])}
# fz = {'adn':allradn[['wak', 'sws']], 'lmn':allrlmn[['wak', 'sws']]}
angs = {'adn':allradn['ang'], 'lmn':allrlmn['ang']}

zbins = np.linspace(-1.5, 1.5, 100)

angbins = np.linspace(0, np.pi, 4)

##############################################
# Mean cc
##############################################
meancc = {}

for e in ['wak', 'sws']:

    meancc[e] = {}

    for i, g in enumerate(['adn', 'lmn']):
    
        groups = angs[g].groupby(np.digitize(angs[g], angbins))
            
        for j in range(angbins.shape[0]-1):

            meancc[e][g+'-'+str(j)] = allcc[g][e][groups.groups[j+1]].mean(1)

    meancc[e] = pd.DataFrame.from_dict(meancc[e])
    # meancc[e] = meancc[e].rolling(15, win_type='gaussian').sum(std=2)

##############################################
# Hist of Fisher Z
##############################################
p = {}
meanp = {}

for i, g in enumerate(fz.keys()):

    groups = fz[g].groupby(np.digitize(angs[g], angbins))

    for j in range(angbins.shape[0]-1):
    
        idx = groups.groups[j+1]
        z = fz[g].loc[idx]

        count = np.histogram(np.abs(z['wak'] - z['sws']), zbins)[0]
        count = count/np.sum(count)  

        p[g+'-'+str(j)] = count
        meanp[g+'-'+str(j)] = np.mean(z['wak'] - z['sws'])

p = pd.DataFrame.from_dict(p)
p = p.set_index(pd.Index(zbins[0:-1] + np.diff(zbins)/2))
p = p.rolling(10, win_type='gaussian').sum(std=1)


####
colors = rcParams['axes.prop_cycle'].by_key()['color']

figure()
gs = GridSpec(3,3)

for i in range(angbins.shape[0]-1):

    for j, e in enumerate(['wak', 'sws']):
        subplot(gs[i,j])

        for k, g in enumerate(['adn', 'lmn']):
            plot(meancc[e][g+'-'+str(i)])

        ylim(np.min(meancc[e]), np.max(meancc[e]))

        if i == 0:
            title(e)

    subplot(gs[i,-1])

    for j, g in enumerate(fz.keys()):

        bar(p.index.values, p[g+"-"+str(i)], np.mean(np.diff(zbins)), 
            label=g, alpha = 0.5, color=colors[j])
        # axvline(meanp[g+"-"+str(i)], linewidth=1, color=colors[j])
    
    if i == 0:
        title("$|Z_{wake} - Z_{sws}|$")
        legend()

show()




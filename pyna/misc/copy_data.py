# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-31 14:54:10
# @Last Modified by:   gviejo
# @Last Modified time: 2023-07-23 10:28:59
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PoissonRegressor
from GLM_HMM import GLM_HMM
from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson
from shutil import copytree


############################################################################################### 
# GENERAL infos
###############################################################################################
if os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'

target_dir = "/mnt/Data/Data"

# datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])



for s in datasets:
    print(s)

    mouse = os.path.join(target_dir, os.path.dirname(s))

    if not os.path.exists(mouse):
        os.mkdir(mouse)

    session = os.path.join(target_dir, s)
    if not os.path.exists(session):
        os.mkdir(session)

    nwb = os.path.join(target_dir, s, 'pynapplenwb')
    # if not os.path.exists(nwb):
    #     os.mkdir(nwb)

    path = os.path.join(data_directory, s)
    if os.path.isdir(os.path.join(path, "pynapplenwb")):

        nwb_path = os.path.join(path, "pynapplenwb")
        
        try:
            copytree(nwb_path, nwb)
        except:
            pass


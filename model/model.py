# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-07-13 21:28:56
# @Last Modified by:   gviejo
# @Last Modified time: 2025-07-13 22:14:00

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import jit, njit
import os



def make_direct_weights(N_in, N_out):
    w = np.eye(N_in)
    w = np.repeat(w, N_out//N_in, axis=0)
    return w

def make_circular_weights(N_in, N_out, sigma=10):
    x = np.arange(-N_out//2, N_out//2)
    y = np.exp(-(x * x) / sigma)
    
    # Manual tiling replacement: concatenate y with itself
    y_tiled = np.concatenate((y, y))
    
    # Slice the middle portion
    y = y_tiled[N_out//2:-N_out//2]
    
    w = np.zeros((N_out, N_in))
    for i in range(N_in):
        w[:,i] = np.roll(y, i*N_out//N_in)
    
    return w

@njit
def sigmoide(x, beta=10, thr=1):
    return 1/(1+np.exp(-(x-thr)*beta))



def run_model(
	N_t=1000,,
	tau = 0.1,
	N_lmn = 36,
	N_adn = 360,
	noise_lmn_=1.0, # Set to 0 during wake
	noise_adn_=1.0, # Set to 0 during wake
	noise_trn_=1.0, # Set to 0 during wake
	w_lmn_adn_=1.5,
	w_adn_trn_=1.0,
	w_trn_adn_=0.1,
	w_psb_lmn_=0.02, # OPTO PSB Feedback
	thr_adn=1.0,
	thr_cal=1.0,
	thr_shu=1.0,
	sigma_adn_lmn = 200,
	sigma_psb_lmn = 10,
	D_lmn = 0.8,
	I_lmn = 1.0 # 0 for sleep
	):

	#############################
	# LMN
	#############################
	inp_lmn = np.zeros((N_t, N_lmn))
	x = np.arange(-N_lmn//2, N_lmn//2)
	y = np.exp(-(x * x) / 100)
	for i in range(N_t):
	    inp_lmn[i] = y
	    if i%50 == 0:
	        y = np.roll(y, 1)

	noise_lmn = np.random.randn(N_t, N_lmn)*noise_lmn_
	r_lmn = np.zeros((N_t, N_lmn))
	x_lmn = np.zeros((N_t, N_lmn))


	#############################
	# ADN
	#############################
	w_lmn_adn = make_circular_weights(N_lmn, N_adn, sigma=sigma_adn_lmn)*w_lmn_adn_
	noise_adn =  np.random.randn(N_t, N_adn)*noise_adn_
	noise_trn =  np.random.randn(N_t)*noise_trn_
	r_adn = np.zeros((N_t, N_adn))
	x_adn = np.zeros((N_t, N_adn))
	# x_cal = np.zeros((N_t, N_adn))
	I_ext = np.zeros((N_t, N_adn))


	#############################
	# TRN
	#############################
	r_trn = np.zeros((N_t))
	x_trn = np.zeros((N_t))
	w_adn_trn = w_adn_trn_
	w_trn_adn = w_trn_adn_


	############################
	# PSB FEEDback
	############################
	w_psb_lmn = make_circular_weights(N_adn, N_lmn, sigma=sigma_psb_lmn)*w_psb_lmn_


	###########################
	# MAIN LOOP
	###########################

	for i in range(1, N_t):
    
	    I_lmn = np.dot(w_psb_lmn, r_adn[i-1]) + inp_lmn[i]

	    # LMN
	    x_lmn[i] = x_lmn[i-1] + tau * (
	        -x_lmn[i-1] 
	        + noise_lmn[i]
	        + I_lmn
	        + D_lmn
	        )
	    r_lmn[i] = np.maximum(0, x_lmn[i])

	    # ADN
	    I_ext[i] = np.dot(w_lmn_adn, r_lmn[i]) - r_trn[i-1] * w_trn_adn #+ sigmoide(-x_cal[i], thr=-thr_shu)


	    # Calcium
	    # x_cal[i] = x_cal[i-1] + tau * (
	    #     - x_cal[i-1]
	    #     + sigmoide(r_adn[i-1], thr=thr_cal)        
	    #     )

	    
	    x_adn[i] = x_adn[i-1] + tau * (
	        - x_adn[i-1]
	        + I_ext[i]
	        + noise_adn[i]        
	        )

	    r_adn[i] = sigmoide(x_adn[i], thr=thr_adn)

	    # TRN
	    x_trn[i] = x_trn[i-1] + tau * (
	        -x_trn[i-1]
	        + np.sum(r_adn[i])*w_adn_trn
	        + noise_trn[i]
	        )

	    # r_trn[i] = np.maximum(0, np.tanh(x_trn[i]))
	    r_trn[i] = np.maximum(0, x_trn[i])

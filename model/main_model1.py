# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   gviejo
# @Last Modified time: 2025-06-29 14:32:29
"""
2 LMN -> 4 ADN 
Non linearity + CAN Current + inhibition in ADN

"""

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from scipy.stats import pearsonr

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import jit, njit


def make_LMN_ADN_weights(N_in, N_out):
	w = np.eye(N_in)
	w = np.repeat(w, N_out//N_in, axis=0)
	return w

def make_PSB_LMN_weights(N_in, N_out, sigma=10):
    x = np.arange(-N_in//2, N_in//2)
    y = np.exp(-(x * x) / sigma)
    
    # Manual tiling replacement: concatenate y with itself
    y_tiled = np.concatenate((y, y))
    
    # Slice the middle portion
    y = y_tiled[N_in//2:-N_in//2]
    
    w = np.zeros((N_out, N_in))
    for i in range(N_out):
        w[i] = np.roll(y, i*N_in//N_out)
    
    return w

@njit
def sigmoide(x, beta=30, thr=1):
	return 1/(1+np.exp(-(x-thr)*beta))

# # @njit
# def run_network(w_lmn_lmn, noise_lmn_, 
# 				w_lmn_adn_, noise_adn_, 
# 				w_adn_trn_, w_trn_adn_, thr_adn, N_t=4000):
tau = 0.1
N_lmn = 2
N_adn = 4

noise_lmn_=0.1
noise_adn_=0.1
noise_cal_=0.1

w_lmn_adn_=1
w_adn_trn_=1
w_trn_adn_=1

thr_adn=1.5
thr_cal=1.0
thr_shu=1.0

sigma_adn_lmn = 2

I_lmn = 1.0

N_t=2000



#############################
# LMN
#############################
inp_lmn = np.zeros((N_t, N_lmn))
noise_lmn = np.random.randn(N_t, N_lmn)*noise_lmn_
r_lmn = np.zeros((N_t, N_lmn))
x_lmn = np.zeros((N_t, N_lmn))


#############################
# ADN
#############################
# w_lmn_adn = make_LMN_ADN_weights(N_lmn, N_adn)*w_lmn_adn_
w_lmn_adn = make_PSB_LMN_weights(N_lmn, N_adn, sigma=sigma_adn_lmn)*w_lmn_adn_
noise_adn =  np.random.randn(N_t, N_adn)*noise_adn_
noise_cal =  np.random.randn(N_t, N_adn)*noise_cal_
r_adn = np.zeros((N_t, N_adn))
x_adn = np.zeros((N_t, N_adn))
x_cal = np.zeros((N_t, N_adn))
I_ext = np.zeros((N_t, N_adn))


#############################
# TRN
#############################
r_trn = np.zeros((N_t))
x_trn = np.zeros((N_t))
w_adn_trn = w_adn_trn_
w_trn_adn = w_trn_adn_


###########################
# MAIN LOOP
###########################

for i in range(1, N_t):

	# LMN	
	x_lmn[i] = x_lmn[i-1] + tau * (
		-x_lmn[i-1] 
		+ noise_lmn[i]
		+ I_lmn
		)
	r_lmn[i] = np.maximum(0, x_lmn[i])

	# ADN
	I_ext[i] = np.dot(w_lmn_adn, r_lmn[i]) - r_trn[i-1] * w_trn_adn


	# Calcium
	x_cal[i] = x_cal[i-1] + tau * (
		- x_cal[i-1]
		+ sigmoide(x_adn[i-1], thr=thr_cal)
		+ noise_cal[i]
		)

	
	x_adn[i] = x_adn[i-1] + tau * (
		- x_adn[i-1]
		+ I_ext[i]
		+ noise_adn[i]
		+ sigmoide(-x_cal[i], thr=-thr_shu)
		)

	r_adn[i] = sigmoide(x_adn[i], thr=thr_adn)	

	# TRN
	x_trn[i] = x_trn[i-1] + tau * (
		-x_trn[i-1]
		+ np.sum(r_adn[i])*w_adn_trn
		)

	# r_trn[i] = np.maximum(0, np.tanh(x_trn[i]))
	r_trn[i] = np.maximum(0, x_trn[i])


cmap = get_cmap('coolwarm')


figure()
ax = subplot(511)
plot(r_lmn, '-')
ylabel("r_lmn")
ax = subplot(512, sharex=ax)
for i in range(N_adn):
	plot(x_adn[:,i], '-', color=cmap(i/N_adn))
axhline(thr_adn, linestyle='--')
axhline(thr_cal)
ylabel("x_adn")
subplot(513,sharex=ax)
for i in range(N_adn):
	plot(x_cal[:,i], '-', color=cmap(i/N_adn))
axhline(thr_shu)
ylabel("X_cal")
subplot(514, sharex=ax)
for i in range(N_adn):
	plot(r_adn[:,i], '-', color=cmap(i/N_adn))
ylabel("r_adn")
subplot(515, sharex=ax)
plot(r_trn, '-', color='red')
plot(x_trn, '--', color='gray')
ylabel("r_trn")
tight_layout()


show()






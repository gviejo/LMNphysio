# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   gviejo
# @Last Modified time: 2025-06-29 21:49:43
"""
N LMN -> N ADN 
Non linearity + CAN Current + inhibition in ADN + PSB Feedback

"""

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import jit, njit


def make_direct_weights(N_in, N_out):
	w = np.eye(N_in)
	w = np.repeat(w, N_out//N_in, axis=0)
	return w

def make_circular_weights(N_in, N_out, sigma=10):
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
def sigmoide(x, beta=20, thr=1):
	return 1/(1+np.exp(-(x-thr)*beta))

# # @njit
# def run_network(w_lmn_lmn, noise_lmn_, 
# 				w_lmn_adn_, noise_adn_, 
# 				w_adn_trn_, w_trn_adn_, thr_adn, N_t=4000):
tau = 0.1
N_lmn = 12
N_adn = 48

noise_lmn_=1.0
noise_adn_=0.1
noise_cal_=0.2

w_lmn_adn_=1
w_adn_trn_=1
w_trn_adn_=1
w_psb_lmn_=1

thr_adn=1.0
thr_cal=0.5
thr_shu=1.0

sigma_adn_lmn = 1
sigma_psb_lmn = 8

D_lmn = 1-w_psb_lmn_


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
# w_lmn_adn = make_circular_weights(N_lmn, N_adn, sigma=sigma_adn_lmn)*w_lmn_adn_
w_lmn_adn = make_direct_weights(N_lmn, N_adn)*w_lmn_adn_
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


############################
# PSB FEEDback
############################
w_psb_lmn = make_circular_weights(N_adn, N_lmn, sigma=sigma_psb_lmn)*w_psb_lmn_


###########################
# MAIN LOOP
###########################

for i in range(1, N_t):

	I_lmn = np.dot(w_psb_lmn, r_adn[i-1])

	# LMN
	x_lmn[i] = x_lmn[i-1] + tau * (
		-x_lmn[i-1] 
		+ noise_lmn[i]
		+ I_lmn
		+ D_lmn
		)
	r_lmn[i] = np.maximum(0, x_lmn[i])

	# ADN
	I_ext[i] = np.dot(w_lmn_adn, r_lmn[i]) - r_trn[i-1] * w_trn_adn + sigmoide(-x_cal[i], thr=-thr_shu)


	# Calcium
	x_cal[i] = x_cal[i-1] + tau * (
		- x_cal[i-1]
		+ sigmoide(r_adn[i-1], thr=thr_cal)
		+ noise_cal[i]
		)

	
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
		)

	# r_trn[i] = np.maximum(0, np.tanh(x_trn[i]))
	r_trn[i] = np.maximum(0, x_trn[i])


# idx = np.mean(r_adn, 1) > np.percentile(np.mean(r_adn, 1), 20)

tmp = StandardScaler().fit_transform(gaussian_filter1d(r_adn, 1, axis=0))
imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(tmp)
# imap = Isomap(n_components=2).fit_transform(tmp)
tmp = StandardScaler().fit_transform(gaussian_filter1d(r_lmn, 1, axis=0))
imap2 = KernelPCA(n_components=2, kernel='cosine').fit_transform(tmp)
# imap2 = Isomap(n_components=2).fit_transform(tmp)

# imap = Isomap(n_components=2).fit_transform(tmp)
# from umap import UMAP
# imap = UMAP().fit_transform(r_adn)


figure()
n_rows = 6
ax = subplot(n_rows,1,1)
# plot(r_lmn, '-')
pcolormesh(r_lmn.T, cmap='jet')#, vmin=0.9)
ylabel("r_lmn")
ax = subplot(n_rows,1,2, sharex=ax)
plot(x_adn, '-')
axhline(thr_adn, linestyle='--')
ylabel("x_adn")
subplot(n_rows,1,3,sharex=ax)
plot(x_cal, '-')
axhline(thr_shu)
ylabel("X_cal")
subplot(n_rows,1,4, sharex=ax)
plot(r_adn, '-')
axhline(thr_cal)
ylabel("r_adn")
subplot(n_rows,1,5, sharex=ax)
pcolormesh(r_adn.T, cmap='jet')
ylabel("r_adn")
subplot(n_rows,1,6, sharex=ax)
plot(r_trn, '-', color='red')
plot(x_trn, '--', color='gray')
ylabel("r_trn")

figure()
subplot(121)
scatter(imap2[:,0], imap2[:,1], 1)
title("LMN")
subplot(122)
scatter(imap[:,0], imap[:,1], 1)
title("ADN")



show()






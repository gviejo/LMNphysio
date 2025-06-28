# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   gviejo
# @Last Modified time: 2025-06-28 13:06:26
"""
N LMN -> N ADN 
Non linearity + CAN Current + inhibition in ADN

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


@njit
def sigmoide(x, beta=50, thr=1):
	return 1/(1+np.exp(-(x-thr)*beta))

# # @njit
# def run_network(w_lmn_lmn, noise_lmn_, 
# 				w_lmn_adn_, noise_adn_, 
# 				w_adn_trn_, w_trn_adn_, thr_adn, N_t=4000):
tau = 0.1
N_lmn = 12
N_adn = 12

noise_lmn_=0.1
noise_adn_=0.1
noise_cal_=0.1

w_lmn_adn_=1
w_adn_trn_=1
w_trn_adn_=1

thr_adn=1.5
thr_cal=1.0
thr_shu=1.0

I_lmn = 1.0

N_t=5000



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
w_lmn_adn = w_lmn_adn_
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
	I_ext[i] = r_lmn[i]*w_lmn_adn - r_trn[i-1] * w_trn_adn


	# Calcium
	x_cal[i] = x_cal[i-1] + tau * (
		- x_cal[i-1]
		+ sigmoide(x_adn[i-1], thr=thr_cal, beta=10)		
		+ noise_cal[i]
		)

	
	x_adn[i] = x_adn[i-1] + tau * (
		- x_adn[i-1]
		+ I_ext[i]
		+ noise_adn[i]
		+ sigmoide(-x_cal[i], thr=-thr_shu, beta=10)
		)

	r_adn[i] = sigmoide(x_adn[i], thr=thr_adn)	


	# TRN
	x_trn[i] = x_trn[i-1] + tau * (
		-x_trn[i-1]
		+ np.sum(r_adn[i])*w_adn_trn
		)

	# r_trn[i] = np.maximum(0, np.tanh(x_trn[i]))
	r_trn[i] = np.maximum(0, x_trn[i])



tmp = StandardScaler().fit_transform(gaussian_filter1d(r_adn, 1, axis=0))
idx = np.mean(tmp, 1) > np.percentile(np.mean(tmp, 1), 80)
imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(tmp)
# imap = Isomap(n_components=2).fit_transform(tmp)
# from umap import UMAP
# imap = UMAP().fit_transform(r_adn)


figure()
n_rows = 6
ax = subplot(n_rows,1,1)
# plot(r_lmn, '-')
pcolormesh(r_lmn.T, cmap='jet', vmin=0.9)
ylabel("r_lmn")
ax = subplot(n_rows,1,2, sharex=ax)
plot(x_adn, '-')
axhline(thr_adn, linestyle='--')
axhline(thr_cal)
ylabel("x_adn")
subplot(n_rows,1,3,sharex=ax)
plot(x_cal, '-')
axhline(thr_shu)
ylabel("X_cal")
subplot(n_rows,1,4, sharex=ax)
plot(r_adn, '-')
ylabel("r_adn")
subplot(n_rows,1,5, sharex=ax)
pcolormesh(r_adn.T, cmap='jet')
ylabel("r_adn")
subplot(n_rows,1,6, sharex=ax)
plot(r_trn, '-', color='red')
plot(x_trn, '--', color='gray')
ylabel("r_trn")

figure()
scatter(imap[:,0], imap[:,1])


show()






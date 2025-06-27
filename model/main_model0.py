# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-06-26 20:21:36
# @Last Modified by:   gviejo
# @Last Modified time: 2025-06-27 08:36:21

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from scipy.stats import pearsonr

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import jit, njit

@njit
def sigmoide(x, beta=20, thr=1):
	return 1/(1+np.exp(-(x-thr)*beta))

# # @njit
# def run_network(noise_lmn_, w_lmn_adn_, noise_adn_, 
# 				thr_adn, N_t=4000):
tau = 0.1
N_lmn = 1
N_adn = 1

noise_lmn_=0.2
w_lmn_adn_=1
noise_adn_=0.5



w_adn_trn_=50
w_trn_adn_=50
thr_adn=3
N_t=4000



#############################
# LMN
#############################
inp_lmn = np.zeros((N_t, N_lmn))
noise_lmn = np.random.randn(N_t, N_lmn)*noise_lmn_
r_lmn = np.zeros((N_t, N_lmn))
x_lmn = np.zeros(N_lmn)


#############################
# ADN
#############################
w_lmn_adn = w_lmn_adn_
noise_adn =  np.random.randn(N_t, N_adn)*noise_adn_
r_adn = np.zeros((N_t, N_adn))
# x_adn = np.random.randn(N_adn)
x_adn = np.zeros((N_t, N_adn))
x_cal = np.zeros((N_t, N_adn))
I_ext = np.zeros((N_t, N_adn))


#############################
# TRN
#############################
r_trn = np.zeros((N_t))
x_trn = 0
w_adn_trn = w_adn_trn_
w_trn_adn = w_trn_adn_


###########################
# MAIN LOOP
###########################

for i in range(1, N_t):

	# LMN
	r_lmn[i] = np.maximum(0, np.tanh(x_lmn))
	x_lmn = x_lmn + tau * (
		-x_lmn 
		+ noise_lmn[i]
		+ 1
		)


	# Calcium
	x_cal[i] = x_cal[i-1] + tau * (
		- x_cal[i-1]
		+ sigmoide(x_adn[i-1], thr=1.3)
		+ np.random.randn()
		)

	I_ext[i] = r_lmn[i]*w_lmn_adn

	# I_ext[i] = sigmoide(x_cal+r_lmn[i]*w_lmn_adn-r_trn[i], 10, thr_adn) + noise_adn[i]
	# I_ext[i] = np.maximum(0, np.tanh(x_cal+r_lmn[i]*w_lmn_adn-r_trn[i])) + noise_adn[i]


	x_adn[i] = x_adn[i-1] + tau * (
		- x_adn[i-1]
		+ I_ext[i]
		+ noise_adn[i]
		+ x_cal[i]
		# + (1/(1+np.exp(-(I_ext-thr_adn)*5)))
		# + ( 1/(1+np.exp(-(r_lmn[i]-thr_adn)*5)))*w_lmn_adn
		# + noise_adn[i]
		# - r_trn[i]
		# + x_cal[i]
		)
	# x_trn = x_trn + tau * (
	# 	-x_trn
	# 	+ np.sum(r_adn[i])*w_adn_trn
	# 	)

	# ADN
	r_adn[i] = sigmoide(x_adn[i], thr=1.0)

	# # TRN
	# r_trn[i] = np.maximum(0, np.tanh(x_trn))


	# return (r_lmn, r_adn, r_trn, I_ext)


# N_t = 6000
figure()
ax = subplot(411)
plot(r_lmn, '.-')
title("r_lmn")
ax = subplot(412, sharex=ax)
plot(x_adn, '.-')
axhline(1.1)
axhline(1.0)
title("x_adn")
subplot(413,sharex=ax)
plot(x_cal, '.-')
title("X_cal")
subplot(414, sharex=ax)
plot(r_adn, '.-')
title("r_adn")

show()



# r_lmn, r_adn, r_trn, I_ext = run_network(
# 	w_lmn_lmn=0.0,
# 	noise_lmn_=1,
# 	w_lmn_adn_=1,
# 	noise_adn_=1,
# 	w_adn_trn_=50,
# 	w_trn_adn_=50,
# 	thr_adn=3,
# 	N_t=N_t
# 	)
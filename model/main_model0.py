# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-06-26 20:21:36
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-06-27 17:46:50
"""
First model of the paper 
1 LMN -> 1 ADN 
Non linearity + CAN current

"""

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

noise_lmn_=0.1
noise_adn_=0.1
noise_cal_=0.1

w_lmn_adn_=1

thr_adn=1.5
thr_cal=1.0
thr_shu=1.0

I_lmn = 1.0


N_t=6000



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
	# r_lmn[i] = np.maximum(0, np.tanh(x_lmn))
	# r_lmn[i] = sigmoide(x_lmn[i], beta=1, thr=0)
	r_lmn[i] = np.maximum(0, x_lmn[i])

	# ADN
	I_ext[i] = r_lmn[i]*w_lmn_adn


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

	# # TRN
	# r_trn[i] = np.maximum(0, np.tanh(x_trn))


	# return (r_lmn, r_adn, r_trn, I_ext)


# N_t = 6000
figure()
ax = subplot(411)
plot(r_lmn, '-', label="r_lmn")
plot(x_lmn, '--', label="x_lmn", alpha=0.3)
legend()
title("r_lmn")
ax = subplot(412, sharex=ax)
plot(x_adn, '-')
axhline(thr_adn, linestyle='--')
axhline(thr_cal)
title("x_adn")
subplot(413,sharex=ax)
plot(x_cal, '-')
axhline(thr_shu)
title("X_cal")
subplot(414, sharex=ax)
plot(r_adn, '-')
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
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-06-20 10:28:32
"""
First model of the paper 
LMN -> ADN 
Non linearity + inhibition in ADN

"""

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from scipy.stats import pearsonr

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def make_LMN_weights(N, sigma=100):
	x = np.arange(-N//2, N//2)
	y = np.exp(-(x**2)/sigma)
	y = np.tile(y, 2)[N//2:-N//2]
	w = np.zeros((N,N))
	for i in range(N):
		w[i] = np.roll(y, i)
	return w


tau = 0.1

N_lmn = 60
N_adn = 60
N_psb = 60

N_t = 4000

phase = np.linspace(0, 2*np.pi*(N_t//200), N_t)%(2*np.pi)
idx = np.digitize(phase, np.linspace(0, 2*np.pi, N_lmn))
# phase[N_t//3:] = 0

#############################
# LMN
#############################
inp_lmn = np.zeros((N_t, N_lmn))
inp_lmn[np.arange(N_t),idx] = 1
w_lmn = make_LMN_weights(N_lmn, 30)*0.0
noise_lmn = np.random.randn(N_t, N_lmn)*0.3
r_lmn = np.zeros((N_t, N_lmn))
x_lmn = np.random.randn(N_lmn)


#############################
# ADN
#############################
noise_adn = np.random.randn(N_t, N_adn)*0.01
r_adn = np.zeros((N_t, N_adn))
x_adn = np.random.randn(N_adn)
w_lmn_adn = 10


#############################
# TRN
#############################
r_trn = np.zeros((N_t))
x_trn = np.random.randn()
w_adn_trn = 1


alpha = 1.0
beta = 1.0


###########################
# MAIN LOOP
###########################

for i in range(N_t):

	# REmoving driver	
	if i == N_t//2:
		alpha = 0.0

	# if i == 2*N_t//3:
	# 	beta = 0.0

	# LMN
	r_lmn[i] = np.maximum(0, np.tanh(x_lmn))
	x_lmn = x_lmn + tau * (
		-x_lmn 
		+ np.dot(w_lmn, r_lmn[i])
		# + np.dot(w_psb_lmn, r_psb[i])*beta
		+ noise_lmn[i]
		+ inp_lmn[i]*alpha
		)

	# ADN
	r_adn[i] = np.maximum(0, np.tanh(x_adn))
	# TRN
	r_trn[i] = np.maximum(0, np.tanh(x_trn))

	x_adn = x_adn + tau * (
		-x_adn 
		+ (1/(1+np.exp(-(r_lmn[i]-0.5)*5)))*w_lmn_adn
		+ noise_adn[i]
		- r_trn[i]
		)
	x_trn = x_trn + tau * (
		-x_trn
		+ np.sum(r_adn[i]*w_adn_trn)
		)


imap = {}
popcoh = {}

# for i in range(100,N_t,N_t//3):
# 	imap[i] = {}
for k, r in zip(['lmn', 'adn'],[r_lmn, r_adn]):
	imap[k] = {}
	popcoh[k] = {}
	for i, sl in enumerate([slice(0, N_t//2), slice(N_t//2, N_t)]):
	
		# sum_ = r[sl].sum(1)
		# idx = sum_>np.percentile(sum_, 10)
		tmp = gaussian_filter1d(r[sl], 1)

		imap[k][i] = KernelPCA(n_components=2, kernel='cosine').fit_transform(tmp)

		tmp = np.corrcoef(r[sl].T)
		popcoh[k][i] = tmp[np.triu_indices(tmp.shape[0], 1)]







figure()
subplot(311)
imshow(r_adn.T, aspect='auto')
title("ADN")
subplot(312)
imshow(r_lmn.T, aspect='auto')
title("LMN")
subplot(313)
plot(r_lmn.sum(1), 'o-', label="LMN")
plot(r_adn.sum(1), 'o-', label="ADN")
legend()

figure()
count = 0
for i in range(2):
	for j, k in enumerate(['lmn', 'adn']):
		subplot(2,2,count+1)
		count += 1
		scatter(imap[k][i][:,0], imap[k][i][:,1])
		title(k)
		if i == 0:
			ylabel(['wake', 'sleep'][i])


figure()
subplot(121)
plot(popcoh['lmn'][0], popcoh['lmn'][1], 'o')
title("r="+str(np.round(pearsonr(popcoh['lmn'][0], popcoh['lmn'][1])[0], 3)))
subplot(122)
plot(popcoh['adn'][0], popcoh['adn'][1], 'o')
title("r="+str(np.round(pearsonr(popcoh['adn'][0], popcoh['adn'][1])[0], 3)))
show()













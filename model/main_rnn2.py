# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-02-25 21:25:43
# @Last Modified by:   gviejo
# @Last Modified time: 2023-03-05 22:45:30
import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import njit
from scipy.ndimage import gaussian_filter



def make_LMN_weights(N, sigma=100):
	x = np.arange(-N//2, N//2)
	y = np.exp(-(x**2)/sigma)
	y = np.tile(y, 2)[N//2:-N//2]
	w = np.zeros((N,N))
	for i in range(N):
		w[i] = np.roll(y, i)
	return w

def run_network(parameters, N_t):

	N_lmn = 60
	N_adn = 60
	N_psb = 120

	phase = np.linspace(0, 2*np.pi*12, N_t)%(2*np.pi)
	# phase[N_t//2:] = 0
	idx = np.digitize(phase, np.linspace(0, 2*np.pi, N_lmn))
	inp_lmn = np.zeros((N_t, N_lmn))
	inp_lmn[np.arange(N_t),idx] = 20
	# inp_lmn[N_t//2:,:] = 0
	inp_lmn = gaussian_filter(inp_lmn, (5, 10))

	#############################
	# LMN
	#############################
	w_lmn = make_LMN_weights(N_lmn, parameters['w_sigma'])*parameters['w_lmn_lmn']
	w_lmn[np.diag_indices_from(w_lmn)] = 0.0	
	noise_lmn = np.random.randn(N_t, N_lmn)*parameters['noise_lmn']
	r_lmn = np.zeros((N_t, N_lmn))
	# x_lmn = np.random.randn(N_lmn)
	x_lmn = np.zeros(N_lmn)

	r_lmn[0,30] = 1.0
	x_lmn[30] = 1.0

	#############################
	# ADN
	#############################
	w_lmn_adn = parameters['w_lmn_adn']
	noise_adn = np.random.randn(N_t, N_adn)*parameters['noise_adn']
	r_adn = np.zeros((N_t, N_adn))
	# x_adn = np.random.randn(N_adn)
	x_adn = np.zeros(N_adn)


	#############################
	# PSB
	#############################
	w_adn_psb = np.vstack((
		np.eye(N_adn)*parameters['w_adn_psb'],
		np.zeros((N_psb-N_adn,N_adn))
		))	
	w_psb_psb = np.abs(np.tanh(np.random.randn(N_psb, N_psb)*2.0)*parameters['w_psb_psb'])*-1.0
	w_psb_lmn = (w_adn_psb.copy().T/parameters['w_adn_psb'])*parameters['w_psb_lmn']
	noise_psb = np.random.randn(N_t, N_psb)*parameters['noise_psb']
	r_psb = np.zeros((N_t, N_psb))
	# x_psb = np.random.randn(N_psb)
	x_psb = np.zeros(N_psb)

	#############################
	# TRN
	#############################
	r_trn = np.zeros((N_t))
	x_trn = 0.0
	w_adn_trn = parameters['w_adn_trn']
	w_trn_adn = parameters['w_trn_adn']

	###########################
	# MAIN LOOP
	###########################

	for i in range(N_t):		
		# LMN
		r_lmn[i] = np.maximum(0, np.tanh(x_lmn))
		x_lmn = x_lmn + 0.02 * (
			-x_lmn 
			+ np.dot(w_lmn, r_lmn[i])
			+ np.dot(w_psb_lmn, x_psb)
			+ noise_lmn[i]
			# + inp_lmn[i]
			)
		# ADN / TRN
		r_adn[i] = np.maximum(0, np.tanh(x_adn))
		r_trn[i] = np.maximum(0, np.tanh(x_trn))
		x_adn = x_adn + 0.02 * (
			-x_adn 
			+ (1/(1+np.exp(-(r_lmn[i]-parameters['thr_adn'])*100)))*w_lmn_adn
			+ noise_adn[i]
			- r_trn[i]*w_trn_adn
			)	
		x_trn = x_trn + 0.02 * (
			-x_trn
			+ np.sum(r_adn[i]*w_adn_trn)
			)

		# PSB
		r_psb[i] = np.maximum(0, np.tanh(x_psb))
		x_psb = x_psb + 0.02 * (
			- x_psb 
			+ np.dot(w_adn_psb, r_adn[i])
			+ np.dot(w_psb_psb, r_psb[i])
			+ noise_psb[i]
			)

	return (r_lmn, r_adn, r_psb, r_trn)


# parameters = {
# 	'w_lmn_lmn':0.1,
# 	'w_sigma':15,
# 	'noise_lmn':1.0,
# 	'w_lmn_adn':1.0,
# 	'noise_adn':0.1,
# 	'w_adn_psb':0.5,
# 	'w_psb_lmn':1.0,
# 	'w_psb_psb':0.001,
# 	'noise_psb':1.0,
# 	'w_adn_trn':0.1,
# 	'w_trn_adn':0.9,
# 	'thr_adn':0.2
# }

parameters = {
	'w_lmn_lmn':0.1,
	'w_sigma':15,
	'noise_lmn':1.0,
	'w_lmn_adn':1.0,
	'noise_adn':0.001,
	'w_adn_psb':0.5,
	'w_psb_lmn':1.0,
	'w_psb_psb':0.003,
	'noise_psb':1.0,
	'w_adn_trn':0.1,
	'w_trn_adn':0.9,
	'thr_adn':0.2
}




N_t = 200000


r_lmn, r_adn, r_psb, r_trn = run_network(parameters, N_t)



# tmp = gaussian_filter(r_adn.copy(), (1, 1))
tmp = r_adn.copy()
tmp -= tmp.mean(0)
tmp /= tmp.std(0)
tmp = gaussian_filter(tmp, (2, 2))
r_adn2 = []
step = 20
for i in np.arange(0, N_t, step):
	r_adn2.append(tmp[i:i+step].mean(0))
r_adn2 = np.array(r_adn2)

rsum = r_adn2.sum(1)


imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(r_adn2[rsum>0])
# imap = Isomap(n_components=2).fit_transform(r_adn2)

imap2 = np.histogram2d(imap[:,0], imap[:,1], (30, 30))


figure()
gs = GridSpec(4, 2)
subplot(gs[0,0])
imshow(gaussian_filter(r_lmn.T[:,0:], (1, 1)), aspect='auto')
ylabel("LMN")
subplot(gs[1,0])
imshow(gaussian_filter(r_adn.T[:,0:], (1, 1)), aspect='auto')	
ylabel("ADN")
subplot(gs[2,0])
imshow(gaussian_filter(r_psb.T[:,0:], (1, 1)), aspect='auto')
ylabel("PSB")
subplot(gs[3,0])
# axhline(parameters['thr_adn'], color = 'red')
plot(r_lmn[:,30], label = 'lmn')
plot(r_adn[:,30], label = 'adn')
plot(r_psb[:,30], label = 'psb')
plot(r_trn, label = 'trn')
legend()
xlim(0, N_t)
subplot(gs[0:2,1])
plot(imap[:,0], imap[:,1], 'o')
subplot(gs[2:,1])
imshow(imap2[0])
show()
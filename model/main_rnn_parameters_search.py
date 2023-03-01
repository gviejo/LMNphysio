# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-02-25 21:25:43
# @Last Modified by:   gviejo
# @Last Modified time: 2023-02-28 23:08:57
import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import njit



def make_LMN_weights(N, sigma=100):
	x = np.arange(-N//2, N//2)
	y = np.exp(-(x**2)/sigma)
	y = np.tile(y, 2)[N//2:-N//2]
	w = np.zeros((N,N))
	for i in range(N):
		w[i] = np.roll(y, i)
	return w

def run_network(parameters, N_t):

	N_lmn = 120
	N_adn = 120
	N_psb = 120

	#############################
	# LMN
	#############################
	w_lmn = make_LMN_weights(N_lmn, 30)*parameters['w_lmn_lmn']
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
	w_adn_psb = make_LMN_weights(N_lmn, 1)*parameters['w_adn_psb']
	w_psb_psb = np.random.randn(N_psb, N_psb)*parameters['w_psb_psb']
	w_psb_lmn = make_LMN_weights(N_lmn, 1)*parameters['w_psb_lmn']
	noise_psb = np.random.randn(N_t, N_psb)*parameters['noise_psb']
	r_psb = np.zeros((N_t, N_psb))
	# x_psb = np.random.randn(N_psb)
	x_psb = np.zeros(N_psb)

	#############################
	# TRN
	#############################
	r_trn = np.zeros((N_t))
	x_trn = np.random.randn()
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
		# print((1/(1+np.exp(-(r_lmn[i]-parameters['thr_adn'])*10)))*w_lmn_adn)
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

	return (r_lmn, r_adn, r_psb)


parameters = {
	'w_lmn_lmn':0.01,
	'noise_lmn':1.0,
	'w_lmn_adn':1.0,
	'noise_adn':1.0,
	'w_adn_psb':2.0,
	'w_psb_lmn':1.0,
	'w_psb_psb':0.01,
	'noise_psb':1.0,
	'w_adn_trn':0.5,
	'w_trn_adn':1.0,
	'thr_adn':0.5
}

N_t = 10000

# ############################
# opt = []
# obj = []

# for i in range(2000):
# 	print(i)

# 	# p = {
# 	# 	'w_lmn_lmn':np.random.uniform(0.0, 10.0, 1)[0],
# 	# 	'noise_lmn':np.random.uniform(0.0, 1.0, 1)[0],
# 	# 	'w_lmn_adn':np.random.uniform(0.0, 10.0, 1)[0],
# 	# 	'noise_adn':np.random.uniform(0.0, 1.0, 1)[0],
# 	# 	'w_adn_psb':np.random.uniform(0.0, 10.0, 1)[0],
# 	# 	'w_psb_lmn':np.random.uniform(0.0, 100, 1)[0],
# 	# 	'w_psb_psb':np.random.uniform(0.0, 100, 1)[0],
# 	# 	'noise_psb':np.random.uniform(0.0, 1.0, 1)[0],
# 	# 	'w_adn_trn':np.random.uniform(0.0, 10.0, 1)[0],
# 	# 	'w_trn_adn':np.random.uniform(0.0, 10.0, 1)[0],
# 	# 	'thr_adn'  :np.random.uniform(0.0, 1.0, 1)[0],
# 	# }

# 	p = {
# 		'w_lmn_lmn':0.1,
# 		'noise_lmn':0.3,
# 		'w_lmn_adn':np.random.uniform(0.0, 10.0, 1)[0],
# 		'noise_adn':0.01,
# 		'w_adn_psb':2,
# 		'w_psb_lmn':np.random.uniform(0.0, 100, 1)[0],
# 		'w_psb_psb':0.0,
# 		'noise_psb':0.01,
# 		'w_adn_trn':np.random.uniform(0.0, 10.0, 1)[0],
# 		'w_trn_adn':np.random.uniform(0.0, 10.0, 1)[0],
# 		'thr_adn'  :np.random.uniform(0.0, 1.0, 1)[0],
# 	}


# 	r_lmn, r_adn, r_psb = run_network(p, N_t)

# 	imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(r_adn[100:,:])

# 	mind = np.sqrt(np.sum(np.power(imap, 2), 1)).min()
# 	maxd = np.sqrt(np.sum(np.power(imap, 2), 1)).max()
# 	maxmind = mind-maxd
# 	opt.append(pd.DataFrame({i:p}).T)
# 	obj.append(pd.DataFrame({i:{'min':mind, 'max':maxmind}}).T)



# opt = pd.concat(opt, 0)
# obj = pd.concat(obj, 0)


# nadir = obj.max().values

# # nadir = np.array([-0.7, 0.22])

# best = np.argmin(np.sqrt(np.sum(np.power(obj.values - nadir, 2), 1)))

# parameters = dict(opt.loc[best])

############################

r_lmn, r_adn, r_psb = run_network(parameters, N_t)



# r_adn = np.sqrt(r_adn)


imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(r_adn[500:,:])
# imap = Isomap(n_components=2).fit_transform(r_adn[100:,:])




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
legend()
xlim(0, N_t)
subplot(gs[:,1])
plot(imap[:,0], imap[:,1], 'o')

show()
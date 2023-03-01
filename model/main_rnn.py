# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-02-25 21:25:43
# @Last Modified by:   gviejo
# @Last Modified time: 2023-02-26 11:04:56
import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA

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

N_t = 3000


phase = np.linspace(0, 2*np.pi*(N_t//200), N_t)%(2*np.pi)
idx = np.digitize(phase, np.linspace(0, 2*np.pi, N_lmn))
phase[N_t//3:] = 0

#############################
# LMN
#############################
inp_lmn = np.zeros((N_t, N_lmn))
inp_lmn[np.arange(N_t),idx] = 1
w_lmn = make_LMN_weights(N_lmn, 20)*0.1
noise_lmn = np.random.randn(N_t, N_lmn)*0.3
r_lmn = np.zeros((N_t, N_lmn))
x_lmn = np.random.randn(N_lmn)

#############################
# ADN
#############################
noise_adn = np.random.randn(N_t, N_adn)*0.01
r_adn = np.zeros((N_t, N_adn))
x_adn = np.random.randn(N_adn)
w_lmn_adn = 3


#############################
# PSB
#############################
w_adn_psb = make_LMN_weights(N_lmn, 1)*2
w_psb_psb = np.random.randn(N_psb, N_psb)*0.0
w_psb_lmn = make_LMN_weights(N_lmn, 5)*50
noise_psb = np.random.randn(N_t, N_psb)*0.01
r_psb = np.zeros((N_t, N_psb))
x_psb = np.random.randn(N_psb)

#############################
# TRN
#############################
r_trn = np.zeros((N_t))
x_trn = np.random.randn()
w_adn_trn = 0.1


alpha = 1.0
beta = 1.0


###########################
# MAIN LOOP
###########################

for i in range(N_t):
	
	if i == N_t//3:
		alpha = 0.0

	if i == 2*N_t//3:
		beta = 0.0

	# LMN
	r_lmn[i] = np.maximum(0, np.tanh(x_lmn))
	x_lmn = x_lmn + 0.1 * (
		-x_lmn 
		+ np.dot(w_lmn, r_lmn[i])
		+ np.dot(w_psb_lmn, r_psb[i])*beta
		+ noise_lmn[i]
		+ inp_lmn[i]*alpha
		)
	# ADN / TRN
	r_adn[i] = np.maximum(0, np.tanh(x_adn))
	r_trn[i] = np.maximum(0, np.tanh(x_trn))
	x_adn = x_adn + 1 * (
		-x_adn 
		+ (1/(1+np.exp(-(r_lmn[i]-0.5)*10)))*w_lmn_adn
		+ noise_adn[i]
		- r_trn[i]
		)	
	x_trn = x_trn + 1 * (
		-x_trn
		+ np.sum(r_adn[i]*w_adn_trn)
		)

	# PSB
	r_psb[i] = np.maximum(0, np.tanh(x_psb))
	x_psb = x_psb + 1 * (
		- x_psb 
		+ np.dot(w_adn_psb, r_adn[i])
		+ np.dot(w_psb_psb, r_psb[i])
		+ noise_psb[i]
		)







imap = {}
for i in range(100,N_t,N_t//3):
	imap[i] = {}
	for k, r in zip(['lmn', 'adn', 'psb'],[r_lmn, r_adn, r_psb]):
		# imap[i][k] = Isomap(n_components=2).fit_transform(r[i:i+N_t//3])
		imap[i][k] = KernelPCA(n_components=2, kernel='cosine').fit_transform(r[i:i+N_t//3])




figure()
gs = GridSpec(5, 3)
subplot(gs[0,:])
plot(np.sin(phase))
xlim(100, N_t)
axvline(N_t//3)
axvline(2*N_t//3)
subplot(gs[1,:])
imshow(gaussian_filter(r_lmn.T[:,100:], (1, 1)), aspect='auto')
ylabel("LMN")
subplot(gs[2,:])
imshow(gaussian_filter(r_adn.T[:,100:], (1, 1)), aspect='auto')	
ylabel("ADN")
subplot(gs[3,:])
imshow(gaussian_filter(r_psb.T[:,100:], (1, 1)), aspect='auto')
ylabel("PSB")
subplot(gs[4,:])
plot(r_lmn[:,0], label = 'lmn')
plot(r_adn[:,0], label = 'adn')
plot(r_psb[:,0], label = 'psb')
legend()
xlim(100, N_t)


figure()
gs = GridSpec(3,3)
for i, n in enumerate(imap.keys()):
	for j, k in enumerate(imap[n].keys()):
		subplot(gs[j,i])
		plot(imap[n][k][:,0], imap[n][k][:,1], 'o')
		ylabel(k)
show()
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-04 16:45:04
"""
N LMN -> N ADN 
Non linearity + CAN Current + inhibition in ADN + PSB Feedback
Population coherence

"""

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import jit, njit




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



tau = 0.1
N_lmn = 36
N_adn = 360

noise_lmn_=0.5
noise_adn_=0.4
noise_trn_=0.4

w_lmn_adn_=1
w_adn_trn_=1
w_trn_adn_=0.9
w_psb_lmn_=0.1

thr_adn=1.0
thr_cal=1.0
thr_shu=1.0

sigma_adn_lmn = 100
sigma_psb_lmn = 10


D_lmn = 0.9

N_t=6000


alpha = 5.0 # Wakefulness -> Sleep
beta = 1.0 # OPTO PSB Feedback


phase = np.linspace(0, 2*np.pi*(N_t//100), N_t)%(2*np.pi)
idx = np.digitize(phase, np.linspace(0, 2*np.pi, N_lmn+1))
if idx.min() == 1:
    idx -= 1

offset = 100
duration = N_t//3
slices = [
    slice(offset, duration*1), #Wakefulness
    slice(duration*1+offset, duration*2), # Full model
    slice(duration*2+offset, duration*3), # Without PSB Feedback            
        ]

#############################
# LMN
#############################
inp_lmn = np.zeros((N_t, N_lmn))
for i in range(N_t):
    inp_lmn[i,idx[i]] = 1.0
# inp_lmn = gaussian_filter1d(inp_lmn, sigma=5, axis=0)

noise_lmn = np.random.randn(N_t, N_lmn)*noise_lmn_
# noise_lmn[0:duration] *= 0.0
r_lmn = np.zeros((N_t, N_lmn))
x_lmn = np.zeros((N_t, N_lmn))


#############################
# ADN
#############################
w_lmn_adn = make_circular_weights(N_lmn, N_adn, sigma=sigma_adn_lmn)*w_lmn_adn_
# w_lmn_adn = make_direct_weights(N_lmn, N_adn)*w_lmn_adn_
noise_adn =  np.random.randn(N_t, N_adn)*noise_adn_
noise_trn =  np.random.randn(N_t)*noise_trn_
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

    if i == slices[0].stop:
        alpha = 0.0        
    if i == slices[1].stop:
        beta = 0.0        


    I_lmn = np.dot(w_psb_lmn, r_adn[i-1]) + inp_lmn[i] * alpha

    # LMN
    x_lmn[i] = x_lmn[i-1] + tau * (
        -x_lmn[i-1] 
        + noise_lmn[i]
        + I_lmn * beta
        + D_lmn
        )
    r_lmn[i] = np.maximum(0, x_lmn[i])

    # ADN
    I_ext[i] = np.dot(w_lmn_adn, r_lmn[i]) - r_trn[i-1] * w_trn_adn + sigmoide(-x_cal[i], thr=-thr_shu)


    # Calcium
    x_cal[i] = x_cal[i-1] + tau * (
        - x_cal[i-1]
        + sigmoide(x_adn[i-1], thr=thr_cal)        
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
        + noise_trn[i]
        )

    # r_trn[i] = np.maximum(0, np.tanh(x_trn[i]))
    r_trn[i] = np.maximum(0, x_trn[i])



imap = {}
popcoh = {}


for k, r in zip(['lmn', 'adn'],[r_lmn, r_adn]):
    imap[k] = {}
    popcoh[k] = {}
    for i, sl in enumerate(slices):
    
        # sum_ = r[sl].sum(1)
        # idx = sum_>np.percentile(sum_, 10)
        # tmp = gaussian_filter(r[sl], sigma=1, order=(0,1))
        tmp = r[sl]
        tmp = StandardScaler().fit_transform(tmp)
        # tmp = r[sl]        
        # tmp = tmp[tmp.mean(1) > np.percentile(np.mean(tmp, 1), 50)]

        # imap[k][i] = KernelPCA(n_components=2, kernel='cosine').fit_transform(tmp)

        # tmp = np.corrcoef(r[sl].T)
        tmp = np.corrcoef(tmp.T)
        popcoh[k][i] = tmp[np.triu_indices(tmp.shape[0], 1)]


colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}
figure()
gs = GridSpec(2,2)
for i, st in enumerate(['lmn', 'adn']):
    for j, k in enumerate([1,2]):
        subplot(gs[j,i])
        plot(popcoh[st][0], popcoh[st][k], 'o', color=colors[st])
        r, p = pearsonr(popcoh[st][0], popcoh[st][k])
        m, b = np.polyfit(popcoh[st][0], popcoh[st][k], 1)
        x = np.linspace(popcoh[st][0].min(), popcoh[st][0].max(),5)
        plot(x, x*m + b)

        xlabel("wake")
        if j == 0:
            ylabel("Full sleep")
        if j == 1:
            ylabel("No PSB feedback")
        title(st + " " + f"r={np.round(r,2)}")



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

show()















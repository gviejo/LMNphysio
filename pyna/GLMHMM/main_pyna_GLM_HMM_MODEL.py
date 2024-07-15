# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-07-15 19:07:24
import numpy as np
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson

from GLM_HMM import GLM_HMM
from GLM import HankelGLM, ConvolvedGLM, CorrelationGLM
import os
import pynapple as nap


T = 5000 # number of data/time points
K = 3 # number of latent states
N = 12 # Number of neurons

#######################################
# Generate the data
#######################################
bins = np.linspace(0, 2*np.pi, 61)
alpha = np.digitize(
    gaussian_filter1d(np.cumsum(np.random.randn(T)*0.5), 2)
    %(2*np.pi), bins
    )-1
x = np.linspace(-np.pi, np.pi, len(bins)-1)
tmp = np.roll(np.exp(-(2.0*x)**2), (len(bins)-1)//2)
tc = np.array([np.roll(tmp, i*(len(bins)-1)//N) for i in range(N)]).T
Y = np.random.poisson(tc[alpha]*5)

tcr = np.random.rand(tc.shape[0], N)*np.mean(Y.sum(0)/T)
Yr = np.random.poisson(tcr[alpha]*2)

Y0 = np.zeros_like(Yr)

############################################################################################### 
# HMM GLM
###############################################################################################

bin_size = 0.02
window_size = bin_size*50.0

############################################
print("fitting GLM")
glm = ConvolvedGLM(Y, bin_size, window_size)
glm.fit_scipy()
# glm.fit_sklearn()            

rglm = ConvolvedGLM(Yr, bin_size, window_size)
rglm.fit_scipy()
# rglm.fit_sklearn()

glm0 = ConvolvedGLM(Y0, bin_size, window_size)
glm0.fit_scipy()

glms = (glm0, glm, rglm)

############################################
# MIXING THE SPIKE DATA
############################################
A = np.eye(K) + np.random.rand(K, K)*0.01
A = A/A.sum(1)[:,None]

Z = np.zeros(T*3, dtype='int')
m, n, o = (0, 1, 0)
Yt = [Y[0]]
for i in range(1, T*3):
    Z[i] = np.sum(np.random.rand()>np.cumsum(A[Z[i-1]]))
    if Z[i] == 0:        
        Yt.append(Y0[m])
        m+=1
    elif Z[i] == 1:
        Yt.append(Y[n])
        n+=1
    elif Z[i] == 2:
        Yt.append(Yr[o])
        o+=1

    if m == T: m = 0
    if n == T: n = 0
    if o == T: o = 0

Yt = np.array(Yt)

############################################
# FITTING THE HMM
############################################
bin_size = 0.02
Yt = nap.TsdFrame(t=np.arange(0, len(Yt))*bin_size, d=Yt)


hmm = GLM_HMM(glms)

hmm.fit_transition(Yt)

# Sampling random trajectories to compute score

random_scores = []

for i in range(1000):
    Ar = np.random.rand(K, K)
    Ar = Ar/Ar.sum(1)[:,None]
    Zr = np.zeros(T*3, dtype='int')
    for j in range(1, T*3):
        Zr[j] = np.sum(np.random.rand()>np.cumsum(Ar[Zr[j-1]]))
    random_scores.append(np.sum(Z == Zr)/len(Z))

random_scores = np.array(random_scores)
##################################################################
# FOR FIGURE supp 1
##################################################################

n = 1000

figure()
subplot(211)
imshow(Yt.values[0:n].T, aspect='auto')
subplot(212)
plot(Z)
plot(hmm.Z.d)

show()





datatosave = {
    "W":np.array([glms[i].W for i in range(len(glms))]),
    "scores":hmm.scores,
    "A":A,
    "bestA":hmm.A,
    "Z":Z[0:n],
    "bestZ":hmm.Z[0:n].values,    
    "tc":tc,
    "tcr":tcr,    
    "Yt":Yt.values[0:n],
    "Y":Y[0:100],
    "Yr":Yr[0:100],
    "B":glm.B,
    "random_scores":random_scores,
    "O":hmm.O[0:n]
    }


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
file_name = "DATA_SUPP_FIG_1_HMM_exemple.pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)



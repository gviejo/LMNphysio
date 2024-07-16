# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-07-16 16:50:33
import numpy as np
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
import os
import pynapple as nap

import nemos as nmo

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

basis = nmo.basis.RaisedCosineBasisLog(
    n_basis_funcs=3, shift=True, mode="conv", window_size=int(window_size/bin_size)
)
_, coupling_basis = basis.evaluate_on_grid(int(window_size/bin_size))

mask = np.repeat(1-np.eye(N), 3, axis=0)

glm = nmo.glm.PopulationGLM(
    regularizer=nmo.regularizer.UnRegularized("LBFGS"),
    feature_mask=mask
    )
glm.fit(basis.compute_features(Y), Y)

rglm = nmo.glm.PopulationGLM(
    regularizer=nmo.regularizer.UnRegularized("LBFGS"),
    feature_mask=mask
    )
rglm.fit(basis.compute_features(Yr), Yr)

glm0 = nmo.glm.PopulationGLM(
    regularizer=nmo.regularizer.UnRegularized("LBFGS"),
    feature_mask=mask
    )
glm0.fit(basis.compute_features(Y0), Y0)

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

X = basis.compute_features(Yt)

tokeep = ~np.any(np.isnan(X.d), 1)

X = X[tokeep] # Features
Yt = Yt[tokeep] # Spike counts
Z = Z[tokeep] # State

############################################
# FITTING THE HMM
############################################

# Computing the observation
# self.O = compute_observation(initial_W, self.X, self.Y, self.K)
# self.O = gaussian_filter(self.O, (1, 0))

def compute_observation(glms, X, Y):
    O = []
    for k in range(len(glms)):
        mu = glms[k].predict(X)
        p = poisson.pmf(k=Y, mu=mu)
        p = np.clip(p, 1e-10, 1.0)
        O.append(p.prod(1))
    O = np.array(O).T

    return O

from GLM_HMM import optimize_transition

class GLM_HMM_nemos(object):

    def __init__(self, glms):
        self.K = len(glms)
        self.glms = glms
        self.initial_W = np.array([glms[i].coef_ for i in range(len(glms))])

    def fit_transition(self, X, Yt):
        self.O = compute_observation(self.glms, X, Yt)

        self.scores = []
        As = []
        Zs = []
        Ws = []

        tokeep = ~np.any(np.isnan(self.O), 1)

        self.O = self.O[tokeep]

        self.O = self.O/self.O.sum(1)[:,np.newaxis]

        T = len(self.O)        

        for _ in range(10):
            A, Z, score = optimize_transition((self.K, T, self.O))
            # A, Z, W, score = optimize_intercept((self.K, self.T, self.initial_W, self.X, self.Y))
            self.scores.append(score)
            As.append(A)
            Zs.append(Z)
            try:
                Ws.append(W)
            except:
                Ws.append(self.initial_W)
                        
        max_L = np.array([score.max() for score in self.scores])
        As = np.array(As)
        # Bs = np.array(Bs)

        self.A = As[np.argmax(max_L)]
        self.best_W = Ws[np.argmax(max_L)]

        # self.Z = nap.Tsd(t = Yt.t[],d = Zs[np.argmax(self.max_L)],time_support = ep)
        self.Z = nap.Tsd(t = Yt.t[tokeep], d = Zs[np.argmax(max_L)])

        eps = {}
        for i in range(self.K):
            ep = self.Z.threshold(i-0.5).threshold(i+0.5, "below").time_support
            # ep = ep.drop_short_intervals(0.01)
            eps[i] = ep

        self.eps = eps
        self.W = np.array([self.glms[i].coef_ for i in range(K)])
        # self.O = compute_observation(best_W, self.X, self.Y, self.K)


hmm = GLM_HMM_nemos(glms)

hmm.fit_transition(X, Yt)

#######################################################################
# Sampling random trajectories to compute score
#######################################################################
from numba import njit

@njit
def get_random_scores(Z, K, n=1000):    
    random_scores = np.zeros(n)
    for i in range(n):        
        Ar = np.random.rand(K, K)
        Ar = Ar/Ar.sum(1)[:,None]
        Zr = np.zeros(len(Z), dtype='int')
        for j in range(1, len(Z)):
            Zr[j] = np.sum(np.random.rand()>np.cumsum(Ar[Zr[j-1]]))
        random_scores[i]=np.sum(Z == Zr)/len(Z)
    return random_scores

random_scores = get_random_scores(Z, K, 1)
random_scores = get_random_scores(Z, K, 1000)
##################################################################
# FOR FIGURE supp 1
##################################################################

n = 1000

figure()
imshow(Yt.values[0:n].T, aspect='auto')
show()





datatosave = {
    "W":hmm.W,
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
    "B":coupling_basis,
    "random_scores":random_scores,
    "O":hmm.O[0:n]
    }


dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
file_name = "DATA_SUPP_FIG_1_HMM_exemple_nemos.pickle"

import _pickle as cPickle

with open(os.path.join(dropbox_path, file_name), "wb") as f:
    cPickle.dump(datatosave, f)



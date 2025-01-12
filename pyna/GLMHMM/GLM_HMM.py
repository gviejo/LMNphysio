# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   gviejo
# @Last Modified time: 2025-01-05 16:15:21
import numpy as np
import os, sys
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor
from numba import njit, jit
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
import pynapple as nap


@njit
def forward(A, T, K, O, init):
    alpha = np.zeros((T, K))
    scaling = np.zeros(T)
    alpha[0] = init*O[0]
    scaling[0] = alpha[0].sum()
    alpha[0] = alpha[0]/scaling[0]
    for t in range(1, T):
        # alpha[t] = np.dot(alpha[t-1], A)*B[:,Y[t]]
        alpha[t] = np.dot(alpha[t-1], A)*O[t]
        scaling[t] = alpha[t].sum()
        alpha[t] = alpha[t]/scaling[t]
    return alpha, scaling

@njit
def backward(A, T, K, O, scaling):
    beta = np.zeros((T, K))
    beta[-1] = 1/scaling[-1]
    for t in np.arange(0, T-1)[::-1]:
        # beta[t] = np.dot(A, beta[t+1]*B[:,Y[t+1]])
        beta[t] = np.dot(A, beta[t+1]*O[t+1])
        beta[t] = beta[t]/scaling[t]
    return beta

def compute_observation(W, X, Y, K):
    O = []
    for k in range(K):
        mu = np.exp(np.einsum('tnk,kn->tn', X, W[k]))        
        # if k == 0 and K>2:
        #     mu*=1e-2
        p = poisson.pmf(k=Y, mu=mu)
        p = np.clip(p, 1e-15, 1.0)
        O.append(p.prod(1))
    O = np.array(O).T

    return O

def loss_all(W, X, Y):
    b = W.reshape(X.shape[2], X.shape[1])
    Xb = np.einsum('tnk,kn->tn', X, b)
    exp_Xb = np.exp(Xb)
    loss = np.sum(exp_Xb, 0) - np.sum(Y*Xb, 0)
    l2 = 0.5*Y.shape[0]*np.sum(np.power(b, 2), 0)            
    grad = np.einsum('tnk,tn->kn', X, exp_Xb - Y) + Y.shape[0]*b
    return np.sum(loss + l2), grad.flatten()

def loss_bias(b, X, Y, W):
    # b = W.reshape(X.shape[2], X.shape[1])
    B = np.vstack((W, b))
    Xb = np.einsum('tnk,kn->tn', X, B)
    exp_Xb = np.exp(Xb)
    loss = np.sum(exp_Xb, 0) - np.sum(Y*Xb, 0)
    l2 = 0.5*Y.shape[0]*np.sum(np.power(b, 2), 0)            
    # grad = np.einsum('tnk,tn->kn', X, exp_Xb - Y) + Y.shape[0]*b
    return np.sum(loss + l2)#, grad.flatten()

def optimize_transition(args):
    K, T, O = args
    score = []
    init = np.random.rand(K)
    init = init/init.sum()
    # A = np.eye(K) + np.random.rand(K, K)*0.1
    A = np.random.rand(K, K)
    A = A/A.sum(1)[:,None]
    for i in range(200):
        alpha, scaling = forward(A, T, K, O, init)                
        beta = backward(A, T, K, O, scaling)
        # Expectation
        E = np.tile(A, (T-1, 1, 1)) * alpha[0:-1,:,None]*beta[1:,None,:]
        # E = E * np.tile(B[:,Y[1:]].T[:,None,:], (1, K, 1)) # Adding emission    
        E = E * np.tile(O[1:][:,None,:], (1, K, 1)) # Adding emission    
        G = np.zeros((T, K))
        G[0:-1] = E.sum(-1)
        G[-1] = alpha[-1]

        # Maximisation
        init = G[0]
        A = E.sum(0)/(G[0:-1].sum(0)[:,None])

        # for j, o in enumerate(np.unique(Y)):
        #     B[:,j] = G[Y == o].sum(0)/G.sum(0)
        print(np.sum(np.log(scaling)))
        score.append(np.sum(np.log(scaling)))

        # if i > 2:
        #     if np.abs(score[-2]-score[-1]) < 1e-20:
        #         break        

    Z = np.argmax(G, 1)

    return A, Z, np.array(score)

def optimize_transition2(args):
    K, T, W, X, Y = args
    scores = []    
    init = np.random.rand(K)
    init = init/init.sum()
    A = np.eye(K) + np.random.rand(K, K)*0.1
    A = A/A.sum(1)[:,None]

    
    for i in range(2):

        # Computing the observation
        O = compute_observation(W, X, Y, K)
        score = [0, 1]

        while np.abs(score[-2]-score[-1]) > 1e-20:
            # Forward/backward
            alpha, scaling = forward(A, T, K, O, init)                
            beta = backward(A, T, K, O, scaling)
            # Expectation
            E = np.tile(A, (T-1, 1, 1)) * alpha[0:-1,:,None]*beta[1:,None,:]
            # E = E * np.tile(B[:,Y[1:]].T[:,None,:], (1, K, 1)) # Adding emission    
            E = E * np.tile(O[1:][:,None,:], (1, K, 1)) # Adding emission    
            G = np.zeros((T, K))
            G[0:-1] = E.sum(-1)
            G[-1] = alpha[-1]

            # Maximisation
            init = G[0]
            A = E.sum(0)/(G[0:-1].sum(0)[:,None])        
            print(i, np.sum(np.log(scaling)))
            score.append(np.sum(np.log(scaling)))

        scores.append(np.array(score[2:]))

        # Learning GLM based on best sequence for each state
        print("ReFitting GLM")
        z = np.argmax(G, 1)
        b0 = np.zeros((W.shape[0], W.shape[2]))
        if K>2:
            rnge = range(1, K)
        else:
            rnge = range(0, K)
        for j in rnge: 
            if np.sum(z == j) > 100:
                solver = minimize(loss_bias, b0[j], args = (X[z==j], Y[z==j], W[j,0:-1,:]), jac=False, method='L-BFGS-B')
                W[j,-1,:] = solver['x'].flatten()


    Z = np.argmax(G, 1)

    return A, Z, W, np.hstack(scores)

def optimize_observation(args):
    K, T, W, X, Y = args
    score = []
    init = np.random.rand(K)
    init = init/init.sum()
    A = np.eye(K) + np.random.rand(K, K)*0.1
    A = A/A.sum(1)[:,None]

    # Computing the observation
    O = compute_observation(W, X, Y, K)
    
    for i in range(100):

        # Forward/backward
        alpha, scaling = forward(A, T, K, O, init)                
        beta = backward(A, T, K, O, scaling)
        # Expectation
        E = np.tile(A, (T-1, 1, 1)) * alpha[0:-1,:,None]*beta[1:,None,:]
        # E = E * np.tile(B[:,Y[1:]].T[:,None,:], (1, K, 1)) # Adding emission    
        E = E * np.tile(O[1:][:,None,:], (1, K, 1)) # Adding emission    
        G = np.zeros((T, K))
        G[0:-1] = E.sum(-1)
        G[-1] = alpha[-1]

        # Maximisation
        init = G[0]
        # A = E.sum(0)/(G[0:-1].sum(0)[:,None])        

        # Learning GLM based on best sequence for each state
        # if i %20 == 0:
        z = np.argmax(G, 1)
        W0 = np.zeros_like(W)
        for j in range(1, K): # NOt learning the glm 0
            if np.sum(z == j) > 100:
                solver = minimize(
                    loss_all, 
                    W0[j].flatten(), 
                    (
                        X[z==j], 
                        Y[z==j]
                        ), 
                    jac=True, method='L-BFGS-B'
                    )
                W0[j] = solver['x'].reshape(W0[j].shape)

        W = W0
        O = compute_observation(W, X, Y, K)

        O = gaussian_filter1d(O, 2, axis=0)

        print(i, np.sum(np.log(scaling)))
        score.append(np.sum(np.log(scaling)))

        if i > 2:
            if np.abs(score[-2]-score[-1]) < 1e-15:
                break

    Z = np.argmax(G, 1)

    return A, Z, W, np.array(score)

def optimize_intercept(args):
    K, T, W, X, Y = args
    scores = []    
    init = np.random.rand(K)
    init = init/init.sum()    
    A = np.eye(K) + np.random.rand(K, K)*0.01
    A = A/A.sum(1)[:,None]
    
    for i in range(1000):

        # Computing the observation
        O = compute_observation(W, X, Y, K)

        # Forward/backward
        alpha, scaling = forward(A, T, K, O, init)                
        beta = backward(A, T, K, O, scaling)
        # Expectation
        E = np.tile(A, (T-1, 1, 1)) * alpha[0:-1,:,None]*beta[1:,None,:]
        E = E * np.tile(O[1:][:,None,:], (1, K, 1)) # Adding emission
        G = np.zeros((T, K))
        G[0:-1] = E.sum(-1)
        G[-1] = alpha[-1]

        # Maximisation
        init = G[0]
        # A = E.sum(0)/(G[0:-1].sum(0)[:,None])

        print(i, np.sum(np.log(scaling)))
        scores.append(np.sum(np.log(scaling)))

        # Learning GLM based on best sequence for each state
        print("ReFitting GLM")
        z = np.argmax(G, 1)
        b0 = np.zeros((W.shape[0], W.shape[2]))        
        for j in range(0, K): 
            if np.sum(z == j) > 100:
                solver = minimize(loss_bias, b0[j], args = (X[z==j], Y[z==j], W[j,0:-1,:]), jac=False, method='L-BFGS-B')
                W[j,-1,:] = solver['x'].flatten()

    Z = np.argmax(G, 1)

    return A, Z, W, np.hstack(scores)    

def fit_transition(K, T, O):    
    score = []
    init = np.random.rand(K)
    init = init/init.sum()
    # A = np.eye(K) + np.random.rand(K, K)*0.1
    # A = np.random.rand(K, K)
    # A = A/A.sum(1)[:,None]    
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def func(a, T, K, O, init):
        p = sigmoid(a)
        A = np.array([p[0], 1-p[0], 1-p[1], p[1]])
        A = A.reshape(K, K)
        alpha, scaling = forward(A, T, K, O, init)
        return -np.sum(np.log(scaling))

    res = minimize(func, np.random.randn(K), (T, K, O, init), tol=1e-10)

    p = sigmoid(res.x)
    A = np.array([p[0], 1-p[0], 1-p[1], p[1]])
    A = A.reshape(K, K)


@jit(nopython=True, cache=True)
def jitrestrict_with_count(time_array, starts, ends, dtype=np.int64):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.int64)
    count = np.zeros(m, dtype=dtype)

    k = 0
    t = 0
    x = 0

    while ends[k] < time_array[t]:
        k += 1

    while k < m:
        # Outside
        while t < n:
            if time_array[t] >= starts[k]:
                break
            t += 1

        # Inside
        while t < n:
            if time_array[t] > ends[k]:
                k += 1
                break
            else:
                ix[x] = t
                count[k] += 1
                x += 1
            t += 1

        if k == m:
            break
        if t == n:
            break

    return ix[0:x], count


@jit(nopython=True, cache=True)
def jitcount(time_array, starts, ends, bin_size, overlap):
    idx, countin = jitrestrict_with_count(time_array, starts, ends)
    time_array = time_array[idx]

    m = starts.shape[0]

    nb_bins = np.zeros(m, dtype=np.int32)
    for k in range(m):
        if (ends[k] - starts[k]) > bin_size:
            # nb_bins[k] = int(np.ceil((ends[k] + bin_size - starts[k]) / bin_size))
            nb_bins[k] = int(np.ceil((ends[k] + bin_size - starts[k]) / (bin_size-overlap)))
        else:
            nb_bins[k] = 1

    nb = np.sum(nb_bins)
    bins = np.zeros(nb, dtype=np.float64)
    cnt = np.zeros(nb, dtype=np.int64)

    k = 0 #
    t = 0 # Spike
    b = 0 # Bins
    t_0 = 0

    while k < m:
        maxb = b + nb_bins[k]
        maxt = t + countin[k]
        lbound = starts[k]

        while b < maxb:
            xpos = lbound + bin_size / 2
            if xpos > ends[k]:
                break
            else:
                bins[b] = xpos
                rbound = np.round(lbound + bin_size, 9)
                while t < maxt:
                    if time_array[t] < rbound:  # similar to numpy hisrogram
                        cnt[b] += 1                        
                        
                        if time_array[t] > rbound-overlap:
                            t_0 += 1

                        t += 1

                    else:
                        break

                lbound += (bin_size-overlap)
                lbound = np.round(lbound, 9)
                b += 1
                t -= t_0 # Goind backward for t
                t_0 = 0

        t = maxt
        k += 1

    new_time_array = bins[0:b]
    new_data_array = cnt[0:b]

    return (new_time_array, new_data_array)


def overlap_count(spikes, bin_size, overlap, ep):
    Y = []
    starts = ep.start
    ends = ep.end    
    for n in spikes.keys():
        time_array = spikes[n].t

        t, c = jitcount(time_array, starts, ends, bin_size, overlap)

        Y.append(c)

    Y = np.array(Y)
    Y = np.transpose(Y)

    return nap.TsdFrame(t=t, d=Y, columns = spikes.keys())


class GLM_HMM(object):

    def __init__(self, glms):
        self.K = len(glms)
        self.glms = glms        
        self.B = glms[0].B
        self.n_basis = self.B.shape[1]
        self.mask = self.glms[0].mask

    def fit_transition(self, spikes, ep=None, bin_size=None):
        self.spikes = spikes        

        ############################################
        # BINNING
        ############################################
        if isinstance(spikes, nap.TsGroup):
            self.N = len(self.spikes)
            count = self.spikes.count(bin_size, ep)
            self.Y = count.values
            self.T = len(self.Y)
            self.time_idx = count.index.values
        else:
            self.N = spikes.shape[1]
            self.Y = spikes
            self.T = spikes.shape[0]
            self.time_idx = spikes.index.values
                

        ############################################
        # CONVOLVING
        ############################################
        self.C = np.zeros((self.T, self.N, self.n_basis))
        for i in range(self.N):
            for j in range(self.n_basis):
                self.C[:,i,j] = np.convolve(self.Y[:,i], self.B[:,j][::-1], mode='same')

        # # Convolving without the 0 bin
        # B0 = np.copy(self.B)
        # B0[len(B0)//2] = 0
        # C0 = np.zeros((self.T, self.N, self.n_basis))
        # for i in range(self.N):
        #     for j in range(self.n_basis):
        #         C0[:,i,j] = np.convolve(self.Y[:,i], B0[:,j][::-1], mode='same')

        self.X = []
        for i in range(self.N):
            # tmp = np.zeros((self.T, self.N, self.n_basis))
            # tmp[:,0:-1,:] = self.C[:,list(set(list(np.arange(self.N))) - set([i])),:]
            # # adding self 
            # tmp[:,-1,:] = C0[:,i,:]
            # apply mask
            # tmp[:,self.mask[i]==0,:] = 0            
            tmp = self.C[:,list(set(list(np.arange(self.N))) - set([i])),:]
            tmp = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
            # tmp = StandardScaler().fit_transform(tmp)
            tmp = np.hstack((tmp, np.ones((len(tmp), 1))))
            self.X.append(tmp)

        self.X = np.array(self.X)

        self.X = np.transpose(self.X, (1, 0, 2))

        self.initial_W = np.array([self.glms[i].W for i in range(self.K)])

        ############################################
        # FITTING THE HMM
        ############################################

        # Computing the observation
        self.O = compute_observation(self.initial_W, self.X, self.Y, self.K)        
        # self.O = gaussian_filter(self.O, (1, 0))

        self.scores = []
        As = []
        Zs = []
        Ws = []

        # args = [(self.K, self.T, self.initial_W, self.X, self.Y) for i in range(15)]
        # with Pool(5) as pool:
        #     for result in pool.map(optimize_intercept, args):
        #         As.append(result[0])
        #         Zs.append(result[1])
        #         Ws.append(result[2])
        #         self.scores.append(result[3])

        for _ in range(2):
            # A, Z, score = optimize_transition((self.K, self.T, self.O))
            A, Z, W, score = optimize_intercept((self.K, self.T, self.initial_W, self.X, self.Y))
            self.scores.append(score)
            As.append(A)
            Zs.append(Z)
            try:
                Ws.append(W)
            except:
                Ws.append(self.initial_W)
                        
        # self.scores = np.array(self.scores).T
        self.max_L = np.array([score.max() for score in self.scores])
        As = np.array(As)
        # Bs = np.array(Bs)

        self.A = As[np.argmax(self.max_L)]
        self.best_W = Ws[np.argmax(self.max_L)]

        self.Z = nap.Tsd(t = self.time_idx, 
            d = Zs[np.argmax(self.max_L)],
            time_support = ep)

        eps = {}
        for i in range(self.K):
            ep = self.Z.threshold(i-0.5).threshold(i+0.5, "below").time_support
            # ep = ep.drop_short_intervals(0.01)
            eps[i] = ep

        self.eps = eps
        self.W = np.array([self.glms[i].W for i in range(self.K)])
        self.O = compute_observation(self.best_W, self.X, self.Y, self.K)


    def fit_observation(self, spikes, ep, bin_size):        
        self.spikes = spikes

        ############################################
        # BINNING
        ############################################
        count = spikes.count(bin_size, ep)
        self.time_idx = count.index.values

        self.N = len(self.spikes)
        self.Y = count.values
        self.T = len(self.Y)

        ############################################
        # CONVOLVING
        ############################################
        self.C = np.zeros((self.T, self.N, self.n_basis))
        for i in range(self.N):
            for j in range(self.n_basis):
                self.C[:,i,j] = np.convolve(self.Y[:,i], self.B[:,j][::-1], mode='same')

        self.X = []
        for i in range(self.N):
            tmp = self.C[:,list(set(list(np.arange(self.N))) - set([i])),:]
            # apply mask
            # tmp[:,self.mask[i]==0,:] = 0            
            tmp = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
            tmp = StandardScaler().fit_transform(tmp)
            self.X.append(tmp)
        self.X = np.array(self.X)

        self.X = np.transpose(self.X, (1, 0, 2))

        self.initial_W = np.array([self.glms[i].W for i in range(self.K)])



        ############################################
        # FITTING THE HMM
        ############################################

        self.scores = []
        As = []
        Zs = []
        Ws = []

        # args = [(self.K, self.T, self.initial_W, self.X, self.Y) for i in range(1)]
        # with Pool(len(args)) as pool:
        #     for result in pool.map(optimize_observation, args):
        #         As.append(result[0])
        #         Zs.append(result[1])
        #         Ws.append(result[2])
        #         self.scores.append(result[3])

        for _ in range(1):
            args = (self.K, self.T, self.initial_W, self.X, self.Y)
            A, Z, W, score = optimize_observation(args)
            self.scores.append(score)
            As.append(A)
            Zs.append(Z)
            Ws.append(W)


        # self.scores = np.array(self.scores).T
        self.max_L = np.array([score.max() for score in self.scores])
        As = np.array(As)
        # Bs = np.array(Bs)

        self.A = As[np.argmax(self.max_L)]
        # B = Bs[np.argmax(scores[-1])]        

        self.Z = nap.Tsd(t = self.time_idx, 
            d = Zs[np.argmax(self.max_L)],
            time_support = ep)

        for i in range(self.K):
            self.glms[i].W = Ws[np.argmax(self.max_L)][i]

        eps = []
        for i in range(self.K):
            ep = self.Z.threshold(i-0.5).threshold(i+0.5, "below").time_support
            ep = ep.drop_short_intervals(0.05)
            eps.append(ep)

        self.eps = eps
        self.W = np.array([self.glms[i].W for i in range(self.K)])
        self.O = compute_observation(self.W, self.X, self.Y, self.K)

class GLM_HMM_nemos(object):

    def __init__(self, glms, n_basis = 3):
        self.K = len(glms)
        self.glms = glms
        self.initial_W = np.array([glms[i].coef_ for i in range(len(glms))])
        self.n_basis = n_basis

    def fit_transition(self, X, Y):        

        self.O = self.compute_observation(X, Y)

        # self.O = self.O.smooth(0.1)

        self.scores = []
        As = []
        Zs = []
        Ws = []

        tokeep = ~np.any(np.isnan(self.O), 1)

        self.O = self.O[tokeep.values]

        T = len(self.O)


        A = fit_transition(self.K, T, np.asarray(self.O.values))

        for _ in range(2):
            A, Z, score = optimize_transition((self.K, T, np.asarray(self.O.values)))
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
        self.Z = nap.Tsd(t = Y.t[tokeep], d = Zs[np.argmax(max_L)])

        eps = {}
        for i in range(self.K):
            ep = self.Z.threshold(i-0.5).threshold(i+0.5, "below").time_support
            # ep = ep.drop_short_intervals(0.01)
            eps[i] = ep

        self.eps = eps
        self.W = np.array([self.glms[i].coef_ for i in range(self.K)])
        # self.O = compute_observation(best_W, self.X, self.Y, self.K)

    def compute_observation(self, X, Y):
        O = []
        for k in range(self.K):
            mu = self.glms[k].predict(X)
            p = poisson.pmf(k=Y, mu=mu)
            # p = np.clip(p, 1e-10, 1.0)
            O.append(p.prod(1))
        O = np.array(O).T
        O = O/O.sum(1)[:,None]

        O = nap.TsdFrame(t=X.t, d=O, time_support=X.time_support)
        return O




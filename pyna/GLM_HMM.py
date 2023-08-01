# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   gviejo
# @Last Modified time: 2023-07-25 13:43:08
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
from numba import njit
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
        if k == 0:
            mu*=1e-2
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
    A = np.eye(K) + np.random.rand(K, K)
    A = A/A.sum(1)[:,None]
    for i in range(100):
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

        if i > 2:
            if np.abs(score[-2]-score[-1]) < 1e-10:
                break        

    Z = np.argmax(G, 1)

    return A, Z, np.array(score)

def optimize_transition2(args):
    K, T, W, X, Y = args
    score = []
    init = np.random.rand(K)
    init = init/init.sum()
    A = np.eye(K) + np.random.rand(K, K)*0.1
    A = A/A.sum(1)[:,None]

    # Computing the observation
    O = compute_observation(W, X, Y, K)
    
    for i in range(200):

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

        # Learning GLM based on best sequence for each state
        if i %5 == 0:
            z = np.argmax(G, 1)
            b0 = np.zeros((W.shape[0], W.shape[2]))
            for j in range(1, K): # NOt learning the glm 0
                if np.sum(z == j) > 100:
                    solver = minimize(loss_bias, b0[j], args = (X[z==j], Y[z==j], W[j,0:-1,:]), jac=False)#, method='L-BFGS-B')
                    W[j,-1,:] = solver['x'].flatten()

        # W = W0
        O = compute_observation(W, X, Y, K)

        # O = gaussian_filter1d(O, 2, axis=0)

        print(i, np.sum(np.log(scaling)))
        score.append(np.sum(np.log(scaling)))

        if i > 2:
            if np.abs(score[-2]-score[-1]) < 1e-15:
                break

    Z = np.argmax(G, 1)

    return A, Z, W, np.array(score)

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

class GLM_HMM(object):

    def __init__(self, glms):
        self.K = len(glms)
        self.glms = glms        
        self.B = glms[0].B
        self.n_basis = self.B.shape[1]
        self.mask = self.glms[0].mask

    def fit_transition(self, spikes, ep, bin_size):        
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
        O = []
        for k in range(self.K):
            mu = self.glms[k].predict(self.X)
            # if k == 0:
            #     mu*=1e-2
            p = poisson.pmf(k=self.Y, mu=mu)
            p = np.clip(p, 1e-15, 1.0)
            O.append(p.prod(1))
        self.O = np.array(O).T

        # self.O = gaussian_filter(self.O, (1, 0))

        self.scores = []
        As = []
        Zs = []
        Ws = []

        # args = [(self.K, self.T, self.initial_W, self.X, self.Y) for i in range(5)]
        # with Pool(len(args)) as pool:
        #     for result in pool.map(optimize_transition2, args):
        #         As.append(result[0])
        #         Zs.append(result[1])
        #         Ws.append(result[2])
        #         self.scores.append(result[3])


        for _ in range(1):
            # A, Z, score = optimize_transition((self.K, self.T, self.O))
            A, Z, W, score = optimize_transition2((self.K, self.T, self.initial_W, self.X, self.Y))

            self.scores.append(score)
            As.append(A)
            Zs.append(Z)


        # self.scores = np.array(self.scores).T
        self.max_L = np.array([score.max() for score in self.scores])
        As = np.array(As)
        # Bs = np.array(Bs)

        self.A = As[np.argmax(self.max_L)]
        # B = Bs[np.argmax(scores[-1])]        

        self.Z = nap.Tsd(t = self.time_idx, 
            d = Zs[np.argmax(self.max_L)],
            time_support = ep)

        eps = []
        for i in range(self.K):
            ep = self.Z.threshold(i-0.5).threshold(i+0.5, "below").time_support
            ep = ep.drop_short_intervals(0.05)
            eps.append(ep)

        self.eps = eps
        self.W = np.array([self.glms[i].W for i in range(self.K)])


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
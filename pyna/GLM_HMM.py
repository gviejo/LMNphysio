# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-04 17:42:47
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
        p = poisson.pmf(k=Y, mu=mu)
        p = np.clip(p, 1e-9, 1.0)
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

def optimize_transition(args):
    K, T, O = args
    score = []
    init = np.random.rand(K)
    init = init/init.sum()
    A = np.random.rand(K, K) + np.eye(K)
    A = A/A.sum(1)[:,None]
    for i in range(40):                         
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

        score.append(np.sum(np.log(scaling)))

    Z = np.argmax(G, 1)

    return A, Z, score

def optimize_observation(args):
    K, T, W, X, Y = args
    score = []
    init = np.random.rand(K)
    init = init/init.sum()
    A = np.random.rand(K, K) + np.eye(K)
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
        A = E.sum(0)/(G[0:-1].sum(0)[:,None])        

        # Learning GLM based on best sequence for each state
        if i %10 == 0:
            z = np.argmax(G, 1)        
            W0 = np.zeros_like(W)
            for j in range(K):
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

        print(np.sum(np.log(scaling)))
        score.append(np.sum(np.log(scaling)))

    Z = np.argmax(G, 1)

    return A, Z, score

class GLM_HMM(object):

    def __init__(self, glms, n_basis = 5):
        self.K = len(glms)
        self.glms = glms
        self.n_basis = 5
        self.B = glms[0].B

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
                self.C[:,i,j] = np.convolve(self.Y[:,i], self.B[:,j], mode='same')

        self.X = []
        for i in range(self.N):
            tmp = self.C[:,list(set(list(np.arange(self.N))) - set([i])),:]
            tmp = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
            tmp = StandardScaler().fit_transform(tmp)
            self.X.append(tmp)
        self.X = np.array(self.X)

        self.X = np.transpose(self.X, (1, 0, 2))

        ############################################
        # FITTING THE HMM
        ############################################

        # Computing the observation
        O = []
        for k in range(self.K):
            mu = self.glms[k].predict(self.X)
            p = poisson.pmf(k=self.Y, mu=mu)
            p = np.clip(p, 1e-9, 1.0)
            O.append(p.prod(1))
        self.O = np.array(O).T

        # self.O = gaussian_filter(self.O, (1, 0))

        self.scores = []
        As = []
        Zs = []

        args = [(self.K, self.T, self.O) for i in range(15)]

        with Pool(len(args)) as pool:
            for result in pool.map(optimize_transition, args):
                As.append(result[0])
                Zs.append(result[1])
                self.scores.append(result[2])

        '''
        for _ in range(3):
            A, Z, score = fit_hmm(self.K, self.T, self.O)

            self.scores.append(score)
            As.append(A)
            Zs.append(Z)


            score = []
            init = np.random.rand(self.K)
            init = init/init.sum()
            A = np.random.rand(self.K, self.K) + np.eye(self.K)
            A = A/A.sum(1)[:,None]
            
            for i in range(50):
                         
                alpha, scaling = forward(A, self.T, self.K, self.O, init)
                
                beta = backward(A, self.T, self.K, self.O, scaling)

                # Expectation
                E = np.tile(A, (self.T-1, 1, 1)) * alpha[0:-1,:,None]*beta[1:,None,:]
                # E = E * np.tile(B[:,Y[1:]].T[:,None,:], (1, K, 1)) # Adding emission    
                E = E * np.tile(self.O[1:][:,None,:], (1, self.K, 1)) # Adding emission    

                G = np.zeros((self.T, self.K))
                G[0:-1] = E.sum(-1)
                G[-1] = alpha[-1]

                # Maximisation
                init = G[0]
                A = E.sum(0)/(G[0:-1].sum(0)[:,None])

                """
                # Learning GLM based on best sequence for each state
                self.z = np.argmax(G, 1)
                new_W = np.zeros_like(self.W)

                if np.all(np.unique(self.z) == np.arange(self.K)):
                
                    for k in range(self.K):
                        args = []
                        for n in range(self.N):
                            args.append((self.X[self.z==k][:,list(set(np.arange(self.N))-set([n]))], self.Y[self.z==k,n]))
                        
                        with Pool(15) as pool:
                            for n, result in enumerate(pool.map(fit_pop_glm, args)):
                                new_W[k,:,n] = result
                    
                self.W = new_W
                        # print(i, k, n)
                        # model= PoissonRegressor()
                        # model.fit(self.X[z==k][:,list(set(np.arange(self.N))-set([n]))], self.Y[z==k,n])
                        # new_W[k,:,n] = model.coef_
                #     W = np.array(W).T
                #     Ws.append(W)
                
                
                # New observation probablities
                O = []
                for k in range(self.K):
                    mu = np.zeros_like(self.Y)
                    for n in range(self.N):
                        mu[:,n] = np.exp(np.dot(self.X[:,list(set(np.arange(self.N))-set([n]))], self.W[k,:,n]))
                    p = poisson.pmf(k=self.Y, mu=mu)
                    p = np.clip(7p, 1e-9, 1.0)
                    O.append(p.prod(1))
                    # O.append(p.sum(1))

                self.O = np.array(O).T
                """

                # for j, o in enumerate(np.unique(Y)):
                #     B[:,j] = G[Y == o].sum(0)/G.sum(0)

                score.append(np.sum(np.log(scaling)))

                print(i, score[-1])

            self.scores.append(score)
            As.append(A)
            Zs.append(np.argmax(G, 1))
            # Bs.append(B)
            '''

        self.scores = np.array(self.scores).T
        As = np.array(As)
        # Bs = np.array(Bs)

        self.A = As[np.argmax(self.scores[-1])]
        # B = Bs[np.argmax(scores[-1])]        

        self.Z = nap.Tsd(t = self.time_idx, 
            d = Zs[np.argmax(self.scores[-1])], 
            time_support = ep)

        eps = []
        for i in range(self.K):
            ep = self.Z.threshold(i-0.5).threshold(i+0.5, "below").time_support
            eps.append(ep)

        self.eps = eps

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
                self.C[:,i,j] = np.convolve(self.Y[:,i], self.B[:,j], mode='same')

        self.X = []
        for i in range(self.N):
            tmp = self.C[:,list(set(list(np.arange(self.N))) - set([i])),:]
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

        args = [(self.K, self.T, self.initial_W, self.X, self.Y) for i in range(15)]        
        with Pool(len(args)) as pool:
            for result in pool.map(optimize_observation, args):
                As.append(result[0])
                Zs.append(result[1])
                self.scores.append(result[2])

        # for _ in range(5):
        #     args = (self.K, self.T, self.initial_W, self.X, self.Y)
        #     A, Z, score = optimize_observation(args)
        #     self.scores.append(score)
        #     As.append(A)
        #     Zs.append(Z)


        self.scores = np.array(self.scores).T
        As = np.array(As)
        # Bs = np.array(Bs)

        self.A = As[np.argmax(self.scores[-1])]
        # B = Bs[np.argmax(scores[-1])]        

        self.Z = nap.Tsd(t = self.time_idx, 
            d = Zs[np.argmax(self.scores[-1])], 
            time_support = ep)

        eps = []
        for i in range(self.K):
            ep = self.Z.threshold(i-0.5).threshold(i+0.5, "below").time_support
            eps.append(ep)

        self.eps = eps
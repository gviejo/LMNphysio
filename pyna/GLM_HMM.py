# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-03 13:59:06
import numpy as np
import os, sys
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor
from numba import njit
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler


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


class GLM_HMM(object):

    def __init__(self, glms, n_basis = 5):
        self.K = len(glms)
        self.glms = glms
        self.n_basis = 5
        self.B = glms[0].B        

    def fit(self, spikes, ep, bin_size):        
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
    

        self.scores = []
        As = []
        Zs = []

        for _ in range(3):

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
                    p = np.clip(p, 1e-9, 1.0)
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

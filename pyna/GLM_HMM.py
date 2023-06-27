# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   gviejo
# @Last Modified time: 2023-06-26 08:32:34
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

# import neurostatslib as nsl
# from neurostatslib.glm import GLM
# from neurostatslib.basis import RaisedCosineBasis, MSplineBasis

# # Jax import ##########################
# import jax
# import jax.numpy as jnp
# from functools import partial
# import jaxopt
# from jaxopt import ProximalGradient
# from jaxopt.prox import prox_lasso



# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# #######################################

# _CORR1 = jax.vmap(partial(jnp.convolve, mode='same'), (0, None), 0)
# # [[n x t],[p x w]] -> [n x p x (t - w + 1)]
# _CORR2 = jax.vmap(_CORR1, (None, 0), 0)


# def fit_sklearn_glm(X, Y


# def fit_pop_glm(spikes, ep, bin_size, ws=100, nb=5):
#     count = spikes.count(bin_size, ep)
#     # if len(order):
#     #     count = count[order]
    
#     Y = count.values
#     spike_data = jnp.array(count.values.T)

#     N = spike_data.shape[0]     
#     T = len(Y)

#     #######################################
#     # Getting the spike basis function
#     #######################################
#     spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=ws)
#     sim_pts = nsl.sample_points.raised_cosine_log(nb, ws)
#     B = spike_basis.gen_basis_funcs(sim_pts)

#     B = B[-2:]

#     spike_basis = B

#     n_basis_funcs, window_size = spike_basis.shape

#     #######################################
#     # FITTING BOTH GLM
#     #######################################
#     def loss_all(b, X, Y):
#         Xb = jnp.dot(X, b) 
#         exp_Xb = jnp.exp(Xb)
#         loss = jnp.sum(exp_Xb, 0) - jnp.sum(Y*Xb, 0) # + jax.scipy.special.gammaln(Y+1)
#         # penalty = jnp.sqrt(jnp.sum(jnp.maximum(b**2, 0), 0))        
#         # return jnp.mean(loss) + 0.1*jnp.sum(jnp.abs(b))
#         return jnp.mean(loss) + 0.1*jnp.sum(jnp.power(b, 2))
#         # return jnp.mean(loss) + 0.1*jnp.sum(jnp.sqrt(jnp.maximum(0, jnp.sum(b**2.0, 1))))

#     def loss1(b, X, y):
#         Xb = np.dot(X, b) 
#         exp_Xb = np.exp(Xb)
#         loss = np.sum(exp_Xb, 0) - np.sum(y*Xb, 0) # + jax.scipy.special.gammaln(Y+1)
#         grad = np.dot(X.T, exp_Xb - y)
#         return loss + 0.5*np.sum(np.power(b, 2)), grad
#         # return jnp.mean(loss) + 0.1*jnp.sum(jnp.sqrt(jnp.maximum(0, jnp.sum(b**2.0, 1))))
    

#     Ws = []
    
#     n_neurons, _ = spike_data.shape

#     X = _CORR2(jnp.atleast_2d(spike_basis),jnp.atleast_2d(spike_data))
#     X = X.reshape(np.prod(X.shape[0:2]), X.shape[-1])
#     X = X.T
#     X = X - X.mean(0)
#     X = X / X.std(0)
#     X = np.hstack((X, jnp.ones((len(X), 1))))

#     b0 = jnp.zeros(((n_neurons*n_basis_funcs)+1, n_neurons))


#     # solver = jaxopt.GradientDescent(
#     #     fun=loss2, maxiter=20000, acceleration=False, verbose=True, stepsize=0.001
#     #     )
#     # W, state = solver.run(b0, X=X, Y=Y)    
#     # solver = ProximalGradient(
#     #     fun=loss1, maxiter=5000, 
#     #     prox=prox_lasso, verbose=True
#     #     )
#     # W, state = solver.run(b0, hyperparams_prox=0.1, X=X, Y=Y[:,0][:,None])

#     W = []

#     for i in range(n_neurons):
#         print(i)
#         # w = minimize(loss1, b0[:,i], (X, Y[:,i]), jac=True, method='newton-cg', options={"disp":True})
#         # W.append(w['x'])

#         model= PoissonRegressor()
#         model.fit(X, Y[:,i])
#         W.append(model.coef_)


#     W = np.array(W).T
    
#     W2 = W[0:-1].reshape((n_neurons, n_basis_funcs, n_neurons))

#     return (W, W2)

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

    def __init__(self, n_state):#, window_size=100, nb=5):        
        self.K = n_state
        # self.window_size = window_size
        # self.nb = nb

        # spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=window_size)
        # sim_pts = nsl.sample_points.raised_cosine_log(nb, window_size)
        # self.B = spike_basis.gen_basis_funcs(sim_pts)

        # self.B = self.B[-2:]

        # self.spike_basis = self.B
        # self.n_basis_funcs, self.window_size = self.spike_basis.shape

    def fit(self, spikes, ep, bin_size):
        count = spikes.count(bin_size, ep)
        # if order:
        #     count = count[order]

        self.Y = count.values
        self.N = self.Y.shape[1]
        self.T = len(self.Y)

        X = gaussian_filter1d(count.values.astype("float"), sigma=1, order=0)
        X = X - X.mean(0)
        X = X / X.std(0)

        # self.X = []
        # for n in range(self.N):
        #     self.X.append(X[:,list(set(np.arange(self.N))-set([n]))])
        # self.X = np.array(self.X)

        self.X = X

        ############################################
        # FITTING THE HMM
        ############################################
        self.W = np.random.randn(self.K, self.N-1, self.N)*0.1
        self.W[1] *= 0.0


        # X = _CORR2(jnp.atleast_2d(self.spike_basis),jnp.atleast_2d(self.Y.T))
        # X = X.reshape(np.prod(X.shape[0:2]), X.shape[-1])
        # X = X.T
        # X = X - X.mean(0)
        # X = X / X.std(0)
        # self.X = np.hstack((X, jnp.ones((len(X), 1))))

        # Computing the observation
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

        self.scores = []
        As = []
        Zs = []

        for _ in range(1):

            score = []
            init = np.random.rand(self.K)
            init = init/init.sum()
            A = np.random.rand(self.K, self.K) + np.eye(self.K)
            A = A/A.sum(1)[:,None]
            
            for i in range(200):
                         
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

                # Learning GLM based on best sequence for each state
                self.z = np.argmax(G, 1)
                new_W = np.zeros_like(self.W)

                if np.all(np.unique(self.z) == np.arange(self.K)):
                
                    for k in range(self.K):
                        args = []
                        for n in range(self.N):
                            args.append((self.X[self.z==k][:,list(set(np.arange(self.N))-set([n]))], self.Y[self.z==k,n]))
                        
                        with Pool(10) as pool:                        
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
        self.bestZ = Zs[np.argmax(self.scores[-1])]


def fit_pop_glm(arg):
    X, Y = arg
    model= PoissonRegressor()
    model.fit(X, Y)
    return model.coef_

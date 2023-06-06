# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   gviejo
# @Last Modified time: 2023-06-05 22:25:56
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


import neurostatslib as nsl
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis, MSplineBasis

# Jax import ##########################
import jax
import jax.numpy as jnp
from functools import partial
import jaxopt
from jaxopt import ProximalGradient
from jaxopt.prox import prox_lasso



jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
#######################################

_CORR1 = jax.vmap(partial(jnp.convolve, mode='same'), (0, None), 0)
# [[n x t],[p x w]] -> [n x p x (t - w + 1)]
_CORR2 = jax.vmap(_CORR1, (None, 0), 0)



def fit_pop_glm(spikes, ep, bin_size, ws=100, nb=5):
    count = spikes.count(bin_size, ep)
    # if len(order):
    #     count = count[order]
    
    Y = count.values
    spike_data = jnp.array(count.values.T)

    N = spike_data.shape[0]     
    T = len(Y)

    #######################################
    # Getting the spike basis function
    #######################################
    spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=ws)
    sim_pts = nsl.sample_points.raised_cosine_linear(nb, ws)
    B = spike_basis.gen_basis_funcs(sim_pts)

    spike_basis = B

    n_basis_funcs, window_size = spike_basis.shape

    #######################################
    # FITTING BOTH GLM
    #######################################
    def loss_all(b, X, Y):
        Xb = jnp.dot(X, b) 
        exp_Xb = jnp.exp(Xb)
        loss = jnp.sum(exp_Xb, 0) - jnp.sum(Y*Xb, 0) # + jax.scipy.special.gammaln(Y+1)
        # penalty = jnp.sqrt(jnp.sum(jnp.maximum(b**2, 0), 0))        
        # return jnp.mean(loss) + 0.1*jnp.sum(jnp.abs(b))
        return jnp.mean(loss) + 0.1*jnp.sum(jnp.power(b, 2))
        # return jnp.mean(loss) + 0.1*jnp.sum(jnp.sqrt(jnp.maximum(0, jnp.sum(b**2.0, 1))))

    def loss1(b, X, y):
        Xb = np.dot(X, b) 
        exp_Xb = np.exp(Xb)
        loss = np.sum(exp_Xb, 0) - np.sum(y*Xb, 0) # + jax.scipy.special.gammaln(Y+1)
        grad = np.dot(X.T, exp_Xb - y)
        return loss + 0.5*np.sum(np.power(b, 2)), grad
        # return jnp.mean(loss) + 0.1*jnp.sum(jnp.sqrt(jnp.maximum(0, jnp.sum(b**2.0, 1))))
    

    Ws = []
    
    n_neurons, _ = spike_data.shape

    X = _CORR2(jnp.atleast_2d(spike_basis),jnp.atleast_2d(spike_data))
    X = X.reshape(np.prod(X.shape[0:2]), X.shape[-1])
    X = X.T
    X = X - X.mean(0)
    X = X / X.std(0)
    X = np.hstack((X, jnp.ones((len(X), 1))))

    b0 = jnp.zeros(((n_neurons*n_basis_funcs)+1, n_neurons))


    # solver = jaxopt.GradientDescent(
    #     fun=loss2, maxiter=20000, acceleration=False, verbose=True, stepsize=0.001
    #     )
    # W, state = solver.run(b0, X=X, Y=Y)    
    # solver = ProximalGradient(
    #     fun=loss1, maxiter=5000, 
    #     prox=prox_lasso, verbose=True
    #     )
    # W, state = solver.run(b0, hyperparams_prox=0.1, X=X, Y=Y[:,0][:,None])

    W = []

    for i in range(n_neurons):
        print(i)
        # w = minimize(loss1, b0[:,i], (X, Y[:,i]), jac=True, method='newton-cg', options={"disp":True})
        # W.append(w['x'])

        model= PoissonRegressor(verbose=1)
        model.fit(X, Y[:,i])
        W.append(model.coef_)


    W = np.array(W).T
    
    W2 = W[0:-1].reshape((n_neurons, n_basis_funcs, n_neurons))

    return (W, W2)


class GLM_HMM(object):

    def __init__(self, Ws, window_size=100, nb=5):
        self.Ws = Ws
        self.K = len(Ws)
        self.window_size = window_size
        self.nb = nb

        spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=window_size)
        sim_pts = nsl.sample_points.raised_cosine_linear(nb, window_size)
        self.B = spike_basis.gen_basis_funcs(sim_pts)
        self.spike_basis = self.B
        self.n_basis_funcs, self.window_size = self.spike_basis.shape

    def fit(self, spikes, ep, bin_size):
        count = spikes.count(bin_size, ep)
        # if order:
        #     count = count[order]

        self.Y = count.values       
        self.N = self.Y.shape[1]
        self.T = len(self.Y)

        ############################################
        # FITTING THE HMM
        ############################################

        X = _CORR2(jnp.atleast_2d(self.spike_basis),jnp.atleast_2d(self.Y.T))
        X = X.reshape(np.prod(X.shape[0:2]), X.shape[-1])
        X = X.T
        X = X - X.mean(0)
        X = X / X.std(0)
        self.X = np.hstack((X, jnp.ones((len(X), 1))))

        O = []
        for i in range(len(self.Ws)):
            p = poisson.pmf(k=self.Y, mu=np.exp(np.dot(self.X, self.Ws[i])))
            O.append(p.prod(1))
            # O.append(p.sum(1))

        self.O = np.array(O).T

        self.scores = []
        As = []
        Zs = []

        for _ in range(2):

            score = []
            init = np.random.rand(self.K)
            init = init/init.sum()
            A = np.random.rand(self.K, self.K)
            A = A/A.sum(1)[:,None]
            
            for i in range(30):
                            
                # Forward
                alpha = np.zeros((self.T, self.K))
                scaling = np.zeros(self.T)
                alpha[0] = init*self.O[0]
                scaling[0] = alpha[0].sum()
                alpha[0] = alpha[0]/scaling[0]
                for t in range(1, self.T):
                    # alpha[t] = np.dot(alpha[t-1], A)*B[:,Y[t]]
                    alpha[t] = np.dot(alpha[t-1], A)*self.O[t]
                    scaling[t] = alpha[t].sum()
                    alpha[t] = alpha[t]/scaling[t]

                # Backward    
                beta = np.zeros((self.T, self.K))
                beta[-1] = 1/scaling[-1]
                for t in np.arange(0, self.T-1)[::-1]:
                    # beta[t] = np.dot(A, beta[t+1]*B[:,Y[t+1]])
                    beta[t] = np.dot(A, beta[t+1]*self.O[t+1])
                    beta[t] = beta[t]/scaling[t]

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



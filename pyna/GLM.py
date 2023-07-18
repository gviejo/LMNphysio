# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-18 18:55:31
import numpy as np
import pynapple as nap
import os, sys
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.preprocessing import StandardScaler
from scipy.linalg import hankel

from jaxopt import GradientDescent, LBFGS, ScipyMinimize
import jax.numpy as jnp

from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson, gamma, norm
from sklearn.linear_model import PoissonRegressor
from numba import njit
from multiprocessing import Pool
from itertools import combinations
import pandas as pd


def offset_matrix(rate, binsize=0.01, windowsize = 0.1):
    idx1 = -np.arange(0, windowsize + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, windowsize + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))

    # Build the Hankel matrix
    tmp = rate
    n_p = len(idx1)
    n_f = len(idx2)
    pad_tmp = np.pad(tmp, (n_p, n_f))
    offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]        

    return offset_tmp, time_idx

class HankelGLM(object):

    def __init__(self, spikes, binsize, windowsize, ep):
        self.spikes = spikes        
        self.N = len(self.spikes)
       
        count = self.spikes.count(binsize, ep)
        self.Y = count.values
        self.T = len(self.Y)

        rates = count/binsize
        rates = rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
        rates = StandardScaler().fit_transform(rates.values)

        idx1 = -np.arange(0, windowsize + binsize, binsize)[::-1][:-1]
        idx2 = np.arange(0, windowsize + binsize, binsize)[1:]
        time_idx = np.hstack((idx1, np.zeros(1), idx2))
    
        # Build the Hankel matrix
        X = []
        for i in range(self.N):
            tmp = rates[:,i]
            n_p = len(idx1)
            n_f = len(idx2)
            pad_tmp = np.pad(tmp, (n_p, n_f))
            offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]
            X.append(offset_tmp.T)

        X = np.array(X)

        self.X = []
        for i in range(self.N):
            tmp = X[list(set(list(np.arange(self.N))) - set([i]))]
            tmp = tmp.reshape(tmp.shape[0]*tmp.shape[1], tmp.shape[2]).T
            tmp = StandardScaler().fit_transform(tmp)
            self.X.append(tmp)
        self.X = np.array(self.X)

        self.X = np.transpose(self.X, (1, 0, 2))

    def fit_scipy(self):

        W0 = np.zeros((self.X.shape[-1], self.N))

        #######################################                        
        def loss_all(W, X, Y):
            b = W.reshape(X.shape[2], X.shape[1])
            Xb = np.einsum('tnk,kn->tn', X, b)
            exp_Xb = np.exp(Xb)
            loss = np.sum(exp_Xb, 0) - np.sum(Y*Xb, 0)
            l2 = 0.5*Y.shape[0]*np.sum(np.power(b, 2), 0)            
            grad = np.einsum('tnk,tn->kn', X, exp_Xb - Y) + Y.shape[0]*b
            return np.sum(loss + l2), grad.flatten()

        solver = minimize(loss_all, W0.flatten(), (self.X, self.Y), jac=True, method='L-BFGS-B')

        W = solver['x'].reshape(W0.shape)
        return W

    def fit_sklearn(self):
        W = []
        for i in range(self.N):
            model = PoissonRegressor(fit_intercept=False)
            model.fit(self.X[:,i,:], self.Y[:,i])
            W.append(model.coef_)

        W = np.array(W).T
        return W

class ConvolvedGLM(object):

    def __init__(self, spikes, binsize, windowsize, ep):
        self.spikes = spikes
        self.N = len(self.spikes)
        
        # mask
        self.mask = np.ones((self.N, self.N-1), dtype=np.int32)
        maxch = self.spikes._metadata["maxch"].values
        groups = self.spikes._metadata["group"].values
        tmp = np.arange(self.N)
        for i in range(self.N):
            for j, k in zip(range(self.N-1), tmp[tmp!=i]):
                if groups[i]==groups[k]:
                    if maxch[i]==maxch[k]:
                            self.mask[i,j] = 0


        count = self.spikes.count(binsize, ep)
        self.Y = count.values
        self.T = len(self.Y)

        nt = int(windowsize/binsize)
        if nt%2==0: nt += 1

        n_basis_funcs = 2
        # V1
        x = np.logspace(np.log10(np.pi * (n_basis_funcs - 1)), -1, nt) - .1
        shifted_x = x[None, :] - (np.pi * np.arange(n_basis_funcs))[:, None]
        B = .5 * (np.cos(np.clip(shifted_x, -np.pi, np.pi)) + 1)        
        B = B.T[::-1]
        self.B = np.vstack((B[::-1], np.zeros((B.shape[0]-1, B.shape[1]))))
        # V2
        # x = np.arange(0, nt, 1)
        # B = []
        # for i in range(2, n_basis_funcs+2):
        #     B.append(gamma.pdf(x, a=i, scale=2))
        # B = np.array(B).T
        # B = B/B.sum(0)
        # self.B = np.vstack((B[::-1], np.zeros((B.shape[0]-1, B.shape[1]))))
        # V3
        # x = np.arange(-nt//2+1, nt//2+1, 1)
        # B = []
        # for i in range(1, n_basis_funcs+1):
        #     B.append(norm.pdf(x, 0, i))
        # B = np.array(B).T
        # B = B/B.sum(0)
        # self.B = B

        self.C = np.zeros((self.T, self.N, n_basis_funcs))
        for i in range(self.N):
            for j in range(n_basis_funcs):
                self.C[:,i,j] = np.convolve(self.Y[:,i], self.B[:,j][::-1], mode='same')        

        self.X = []
        for i in range(self.N):
            tmp = self.C[:,list(set(list(np.arange(self.N))) - set([i])),:]
            # apply mask
            tmp[:,self.mask[i]==0,:] = 0
            tmp = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
            tmp = StandardScaler().fit_transform(tmp)
            self.X.append(tmp)
        self.X = np.array(self.X)

        self.X = np.transpose(self.X, (1, 0, 2))

    def predict(self, X):
        return np.exp(np.einsum('tnk,kn->tn', X, self.W))

    def fit_scipy(self):

        W0 = np.zeros((self.X.shape[-1], self.N))

        #######################################                        
        def loss_all(W, X, Y):
            b = W.reshape(X.shape[2], X.shape[1])
            Xb = np.einsum('tnk,kn->tn', X, b)
            exp_Xb = np.exp(Xb)
            loss = np.sum(exp_Xb, 0) - np.sum(Y*Xb, 0)
            l2 = 0.5*Y.shape[0]*np.sum(np.power(b, 2), 0)            
            grad = np.einsum('tnk,tn->kn', X, exp_Xb - Y) + Y.shape[0]*b
            return np.sum(loss + l2), grad.flatten()

        solver = minimize(loss_all, W0.flatten(), (self.X, self.Y), jac=True, method='L-BFGS-B')

        self.W = solver['x'].reshape(W0.shape)        

    def fit_sklearn(self):
        W = []
        for i in range(self.N):
            model = PoissonRegressor(fit_intercept=False)
            model.fit(self.X[:,i,:], self.Y[:,i])
            W.append(model.coef_)

        self.W = np.array(W).T        

class CorrelationGLM(object):

    def __init__(self, spikes, basename = ""):
        self.spikes = spikes
        self.N = len(self.spikes)
       
        self.pairs = [basename+"_"+i+"-"+j for i,j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
        

    def fit(self, ep, binsize, windowsize, name=""):

        count = self.spikes.count(binsize, ep)

        self.Y = []
        self.X = []

        for p in self.pairs:
            n, t = np.array(p.split("_")[1].split("-")).astype("int")
            self.Y.append(count[t].values)
            
            offset_tar, self.time_idx = offset_matrix(count[n].values, binsize, windowsize)
            offset_mua, time_idx = offset_matrix(count[list(set(count.columns) - set([n,t]))].values.sum(1), binsize, windowsize)
            tmp = np.hstack((offset_tar, offset_mua))            
            tmp = StandardScaler().fit_transform(tmp)
            self.X.append(tmp)

        self.Y = np.array(self.Y).T
        self.X = np.array(self.X)
        self.X = np.transpose(self.X, (1, 0, 2))

        self.fit_scipy()

        W = self.W.reshape(2, len(time_idx), self.Y.shape[-1])
        corrpai = pd.DataFrame(index=time_idx, data=W[0], columns = self.pairs)
        corrmua = pd.DataFrame(index=time_idx, data=W[1], columns = self.pairs)
        
        return corrpai, corrmua

    def predict(self, X):
        return np.exp(np.einsum('tnk,kn->tn', X, self.W))

    def fit_scipy(self):

        W0 = np.zeros((self.X.shape[-1], self.Y.shape[-1]))

        #######################################                        
        def loss_all(W, X, Y):
            b = W.reshape(X.shape[2], X.shape[1])
            Xb = np.einsum('tnk,kn->tn', X, b)
            exp_Xb = np.exp(Xb)
            loss = np.sum(exp_Xb, 0) - np.sum(Y*Xb, 0)
            l2 = 0.5*Y.shape[0]*np.sum(np.power(b, 2), 0)            
            grad = np.einsum('tnk,tn->kn', X, exp_Xb - Y) + Y.shape[0]*b
            return np.sum(loss + l2), grad.flatten()

        solver = minimize(loss_all, W0.flatten(), (self.X, self.Y), jac=True, method='L-BFGS-B')

        self.W = solver['x'].reshape(W0.shape)

    def fit_sklearn(self):
        W = []
        for i in range(self.N):
            model = PoissonRegressor(fit_intercept=False)
            model.fit(self.X[:,i,:], self.Y[:,i])
            W.append(model.coef_)

        self.W = np.array(W).T        



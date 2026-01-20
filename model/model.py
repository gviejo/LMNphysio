# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-07-13 21:28:56
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-17 10:24:06

import numpy as np
from numba import jit, njit
import os



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
def sigmoide(x, beta=10.0, thr=1.0):
    return 1/(1+np.exp(-(x-thr)*beta))

# tau = 0.1
# N_lmn = 36
# N_adn = 360
# noise_lmn_ = 6.0
# noise_adn_ = 1.0
# noise_trn_ = 1.0
# w_lmn_adn_ = 0.39   # LMN to ADN weight
# w_adn_trn_ = 1.0    # ADN to TRN weight
# w_trn_adn_ = 0.05   # TRN to ADN weight
# beta_adn = 3.0      # ADN non-linearity slope
# thr_adn = 1.0       # ADN non-linearity threshold
# sigma_adn_lmn = 100 # LMN to ADN weight spread
# D_lmn = 0.02        # LMN drive during sleep
#
# w_psb_lmn_ = 0.11    # OPTO PSB Feedback
# sigma_psb_lmn = 100  # PSB to LMN weight spread
# I_lmn = 0.43          # LMN wakefulness drive


class Model:
    tau = 0.1
    N_lmn = 36
    N_adn = 360
    noise_lmn_ = 6.0
    noise_adn_ = 0.1
    noise_trn_ = 0.1
    w_lmn_adn_ = 0.39    # LMN to ADN weight
    w_adn_trn_ = 1.0    # ADN to TRN weight
    w_trn_adn_ = 0.05   # TRN to ADN weight
    beta_adn = 3.0      # ADN non-linearity slope
    thr_adn = 1.0       # ADN non-linearity threshold
    sigma_adn_lmn = 100 # LMN to ADN weight spread
    D_lmn = 0.025        # LMN drive during sleep

    w_psb_lmn_ = 0.11    # OPTO PSB Feedback
    sigma_psb_lmn = 100  # PSB to LMN weight spread
    I_lmn = 0.43         # LMN wakefulness drive


    def __init__(self, N_t=2000, **kwargs):
        self.N_t = N_t

        # --- Override class defaults with kwargs if provided ---
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

        #############################
        # LMN
        #############################
        self.inp_lmn = np.zeros((self.N_t, self.N_lmn))
        # gaussian version
        x = np.arange(-self.N_lmn // 2, self.N_lmn // 2)
        y = np.exp(-(x * x) / 10)

        for i in range(self.N_t):
            self.inp_lmn[i] = y
            if i % 50 == 0:
                y = np.roll(y, 1)

        self.noise_lmn = np.random.randn(self.N_t, self.N_lmn) * self.noise_lmn_
        self.r_lmn = np.zeros((self.N_t, self.N_lmn))
        self.x_lmn = np.zeros((self.N_t, self.N_lmn))

        #############################
        # ADN
        #############################
        self.w_lmn_adn = make_circular_weights(self.N_lmn, self.N_adn, sigma=self.sigma_adn_lmn) * self.w_lmn_adn_
        self.noise_adn = np.random.randn(self.N_t, self.N_adn) * self.noise_adn_
        self.noise_trn = np.random.randn(self.N_t) * self.noise_trn_
        self.r_adn = np.zeros((self.N_t, self.N_adn))
        self.x_adn = np.zeros((self.N_t, self.N_adn))
        self.I_ext = np.zeros((self.N_t, self.N_adn))
        # self.x_cal = np.zeros((self.N_t, self.N_adn))  # Uncomment if needed

        #############################
        # TRN
        #############################
        self.r_trn = np.zeros(self.N_t)
        self.x_trn = np.zeros(self.N_t)
        self.w_adn_trn = self.w_adn_trn_
        self.w_trn_adn = self.w_trn_adn_

        #############################
        # PSB Feedback
        #############################
        self.w_psb_lmn = make_circular_weights(self.N_adn, self.N_lmn, sigma=self.sigma_psb_lmn) * self.w_psb_lmn_


    def run(self):
        ###########################
        # MAIN LOOP
        ###########################

        for i in range(1, self.N_t):            

            # LMN
            self.x_lmn[i] = self.x_lmn[i - 1] + self.tau * (
                -self.x_lmn[i - 1]
                + self.noise_lmn[i]
                + np.dot(self.w_psb_lmn, self.r_adn[i - 1]) # PSB Feedback
                + self.inp_lmn[i] * self.I_lmn # Wakefulness drive
                + self.D_lmn
            )
            self.r_lmn[i] = np.maximum(0, self.x_lmn[i])

            # ADN
            self.I_ext[i] = np.dot(self.w_lmn_adn, self.r_lmn[i]) - self.r_trn[i - 1] * self.w_trn_adn
            # + sigmoide(-self.x_cal[i], thr=-self.thr_shu)  # Uncomment if using calcium feedback

            # self.x_cal[i] = self.x_cal[i - 1] + self.tau * (
            #     -self.x_cal[i - 1]
            #     + sigmoide(self.r_adn[i - 1], thr=self.thr_cal)
            # )

            self.x_adn[i] = self.x_adn[i - 1] + self.tau * (
                -self.x_adn[i - 1]
                + self.I_ext[i]
                + self.noise_adn[i]
            )
            self.r_adn[i] = sigmoide(self.x_adn[i], thr=self.thr_adn, beta=self.beta_adn)

            # TRN
            self.x_trn[i] = self.x_trn[i - 1] + self.tau * (
                -self.x_trn[i - 1]
                + np.sum(self.r_adn[i]) * self.w_adn_trn
                + self.noise_trn[i]
            )
            self.r_trn[i] = np.maximum(0, self.x_trn[i])
            # self.r_trn[i] = np.maximum(0, np.tanh(self.x_trn[i]))  # Optional alternative

    def get_parameters(self):
        params = {
            "tau": self.tau,
            "N_lmn": self.N_lmn,
            "N_adn": self.N_adn,
            "noise_lmn_": self.noise_lmn_,
            "noise_adn_": self.noise_adn_,
            "noise_trn_": self.noise_trn_,
            "w_lmn_adn_": self.w_lmn_adn_,
            "w_adn_trn_": self.w_adn_trn_,
            "w_trn_adn_": self.w_trn_adn_,
            "beta_adn": self.beta_adn,
            "thr_adn": self.thr_adn,
            "w_psb_lmn_": self.w_psb_lmn_,
            "sigma_adn_lmn": self.sigma_adn_lmn,
            "sigma_psb_lmn": self.sigma_psb_lmn,
            "D_lmn": self.D_lmn,
            "I_lmn": self.I_lmn,
        }
        return params
        

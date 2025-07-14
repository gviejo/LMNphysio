# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-07-13 21:28:56
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-14 09:51:33

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
def sigmoide(x, beta=5, thr=1):
    return 1/(1+np.exp(-(x-thr)*beta))



class Model:
    tau = 0.1
    N_lmn = 36
    N_adn = 360
    noise_lmn_ = 1.0  # Set to 0 during wake
    noise_adn_ = 1.0  # Set to 0 during wake
    noise_trn_ = 1.0  # Set to 0 during wake
    w_lmn_adn_ = 1.5
    w_adn_trn_ = 1.0
    w_trn_adn_ = 0.05
    w_psb_lmn_ = 0.02  # OPTO PSB Feedback
    thr_adn = 1.0
    thr_cal = 1.0
    thr_shu = 1.0
    sigma_adn_lmn = 200
    sigma_psb_lmn = 10
    D_lmn = 0.8
    I_lmn = 1.0  # 0 for sleep

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
        x = np.arange(-self.N_lmn // 2, self.N_lmn // 2)
        y = np.exp(-(x * x) / 100)

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
            self.r_adn[i] = sigmoide(self.x_adn[i], thr=self.thr_adn)

            # TRN
            self.x_trn[i] = self.x_trn[i - 1] + self.tau * (
                -self.x_trn[i - 1]
                + np.sum(self.r_adn[i]) * self.w_adn_trn
                + self.noise_trn[i]
            )
            self.r_trn[i] = np.maximum(0, self.x_trn[i])
            # self.r_trn[i] = np.maximum(0, np.tanh(self.x_trn[i]))  # Optional alternative

        

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-16 13:55:12
"""
N LMN -> N ADN 
Non linearity + CAN Current + inhibition in ADN

"""

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import jit, njit
import pandas as pd
from model import Model


m_wake = Model(I_lmn = 1.0, D_lmn=0.1) # Wake
m_wake.run()
m_opto = Model(I_lmn = 0.0, w_psb_lmn_=0.0) # Sleep
m_opto.run()


popcoh = {}
for k in ['adn', 'lmn']:    
	popcoh[k] = {}
	for n, m in zip(['wak', 'opto'], [m_wake, m_opto]):
		tmp = np.corrcoef(getattr(m, f"r_{k}").T)
		popcoh[k][n] = tmp[np.triu_indices(tmp.shape[0], 1)]
	popcoh[k] = pd.DataFrame.from_dict(popcoh[k])
		


figure()
for i, st in enumerate(popcoh.keys()):
	subplot(1,2,i+1)
	plot(popcoh[st]['wak'], popcoh[st]['opto'], 'o')
	r, p = pearsonr(popcoh[st]['wak'], popcoh[st]['opto'])
	m, b = np.polyfit(popcoh[st]['wak'], popcoh[st]['opto'], 1)
	x = np.linspace(popcoh[st]['wak'].min(), popcoh[st]['wak'].max(),5)
	plot(x, x*m + b)

	xlim(-1, 1)
	ylim(-1, 1)
	title(st + f" - r={np.round(r, 2)}")


durations = [2000,  100]

figure(figsize=(14, 8))
gs = GridSpec(1, len(durations), wspace=0.4)

for i, (m, duration) in enumerate(zip([m_wake, m_opto], durations)):
    n_rows = 6
    inner_gs = gs[i].subgridspec(n_rows, 1, hspace=0.4)

    ax0 = subplot(inner_gs[0])
    # ax0.plot(m.r_lmn[:duration], '-')
    ax0.pcolormesh(m.r_lmn[:duration].T, cmap='jet', shading='auto')
    ax0.set_ylabel("r_lmn")
    if i == 0:
        ax0.set_title("Wake")
    elif i == 1:
        ax0.set_title("SWS")
    else:
        ax0.set_title("Opto")

    ax1 = subplot(inner_gs[1], sharex=ax0)
    ax1.plot(m.x_adn[:duration], '-')
    ax1.axhline(m.thr_adn, linestyle='--', color='k')
    ax1.set_ylabel("x_adn")

    ax2 = subplot(inner_gs[2], sharex=ax0)
    ax2.plot(m.r_adn[:duration], '-')
    ax2.set_ylabel("r_adn")

    ax3 = subplot(inner_gs[3], sharex=ax0)
    pcm = ax3.pcolormesh(m.r_adn[:duration].T, cmap='jet', shading='auto')
    ax3.set_ylabel("r_adn")
    # colorbar(pcm, ax=ax3)

    ax4 = subplot(inner_gs[4], sharex=ax0)
    ax4.plot(m.r_trn[:duration], '-', color='red')
    ax4.plot(m.x_trn[:duration], '--', color='gray')
    ax4.set_ylabel("r_trn")

    ax5 = subplot(inner_gs[5], sharex=ax0)
    ax5.plot(m.I_ext[:duration])
    ax5.set_ylabel("I ADN")

show()



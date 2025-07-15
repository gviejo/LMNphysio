# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-15 18:22:33
"""
N LMN -> N ADN 
Non linearity + CAN Current + inhibition in ADN + PSB Feedback
Population coherence

"""

import numpy as np
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numba import jit, njit
import os
import pandas as pd
from model import Model


m_wake = Model(I_lmn = 1.0) # Wake
m_wake.run()
m_sws = Model(I_lmn = 0.0) # Sleep
m_sws.run()
m_opto = Model(I_lmn = 0.0, w_psb_lmn_=0.0) # Opto
m_opto.run()



popcoh = {}
for k in ['adn', 'lmn']:    
	popcoh[k] = {}
	for n, m in zip(['wak', 'sws', 'opto'], [m_wake, m_sws, m_opto]):
		tmp = np.corrcoef(getattr(m, f"r_{k}").T)
		popcoh[k][n] = tmp[np.triu_indices(tmp.shape[0], 1)]
	popcoh[k] = pd.DataFrame.from_dict(popcoh[k])
		

datatosave = {
    "lmn" : {e:m.r_lmn for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    "adn" : {e:m.r_adn for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    "trn" : {e:m.r_trn for e, m in zip(["wak", "sws", "opto"], [m_wake, m_sws, m_opto])},
    "popcoh" : popcoh
}
import _pickle as cPickle
filepath = os.path.join(os.path.expanduser("~") + "/Dropbox/LMNphysio/model/model.pickle")
cPickle.dump(datatosave, open(filepath, 'wb'))


figure(figsize = (12, 5))
gs = GridSpec(1, 2)
for i, st in enumerate(popcoh.keys()):
	gs2 = GridSpecFromSubplotSpec(1, 2, gs[0,i])
	for j, e in enumerate(['sws', 'opto']):
		subplot(gs2[0,j])
		gca().set_aspect("equal")
		plot(popcoh[st]['wak'], popcoh[st][e], 'o')
		r, p = pearsonr(popcoh[st]['wak'], popcoh[st][e])
		m, b = np.polyfit(popcoh[st]['wak'], popcoh[st][e], 1)
		x = np.linspace(popcoh[st]['wak'].min(), popcoh[st]['wak'].max(),5)
		plot(x, x*m + b)

		xlim(-1, 1)
		ylim(-1, 1)
		title(st + f" - r={np.round(r, 2)}")


durations = [2000, 1000, 200]  # For m_wake, m_sws, m_opto

figure(figsize=(14, 8))
gs = GridSpec(1, 3, wspace=0.4)

for i, (m, duration) in enumerate(zip([m_wake, m_sws, m_opto], durations)):
    n_rows = 6
    inner_gs = gs[i].subgridspec(n_rows, 1, hspace=0.4)

    ax0 = subplot(inner_gs[0])
    ax0.plot(m.r_lmn[:duration], '-')
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

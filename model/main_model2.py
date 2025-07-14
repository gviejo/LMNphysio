# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-06-19 15:28:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-14 09:51:43
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


m_wake = Model(I_lmn = 1.0) # Wake
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


m = m_opto

figure()
n_rows = 6
ax = subplot(n_rows,1,1)
plot(m.r_lmn, '-')
# pcolormesh(r_lmn.T, cmap='jet', vmin=0.9)
ylabel("r_lmn")
ax = subplot(n_rows,1,2, sharex=ax)
plot(m.x_adn, '-')
axhline(m.thr_adn, linestyle='--')
ylabel("x_adn")
# subplot(n_rows,1,3,sharex=ax)
# plot(m.x_cal, '-')
# axhline(m.thr_shu)
# ylabel("X_cal")
subplot(n_rows,1,3, sharex=ax)
plot(m.r_adn, '-')
axhline(m.thr_cal)
ylabel("r_adn")
subplot(n_rows,1,4, sharex=ax)
pcolormesh(m.r_adn.T, cmap='jet')
ylabel("r_adn")
subplot(n_rows,1,5, sharex=ax)
plot(m.r_trn, '-', color='red')
plot(m.x_trn, '--', color='gray')
ylabel("r_trn")
subplot(n_rows,1,6,sharex=ax)
plot(m.I_ext)
ylabel("I ADN")



show()






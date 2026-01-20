# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-07-17
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-17
"""
Parameter search for Model
Searching over w_psb_lmn_
Computing population coherence for m_wake and m_sws conditions
"""

import numpy as np
from matplotlib.pyplot import *
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm
from model import Model

np.random.seed(0)

# Parameter range
w_psb_lmn_values = np.linspace(0.025, 0.035, 8)

# Store results
results = []

N_t = 10000

for w_psb in tqdm(w_psb_lmn_values):

    # Run wake condition
    m_wake = Model(I_lmn=1.0, D_lmn=0.1, w_psb_lmn_=w_psb, N_t=N_t)
    m_wake.run()

    # Run sleep condition
    m_sws = Model(I_lmn=0.0, w_psb_lmn_=w_psb, N_t=N_t)
    m_sws.run()

    # Run opto condition (no PSB input)
    m_opto = Model(I_lmn=0.0, w_psb_lmn_=0.0, N_t=N_t)
    m_opto.run()

    # Compute population coherence
    popcoh = {}
    for k in ['adn', 'lmn']:
        popcoh[k] = {}
        for n, m in zip(['wak', 'sws', 'opto'], [m_wake, m_sws, m_opto]):
            tmp = np.corrcoef(getattr(m, f"r_{k}").T)
            popcoh[k][n] = tmp[np.triu_indices(tmp.shape[0], 1)]

    # Compute correlation between wake and sws popcoh
    r_adn, _ = pearsonr(popcoh['adn']['wak'], popcoh['adn']['sws'])
    r_lmn, _ = pearsonr(popcoh['lmn']['wak'], popcoh['lmn']['sws'])

    # Compute correlation between wake and opto popcoh
    r_adn_opto, _ = pearsonr(popcoh['adn']['wak'], popcoh['adn']['opto'])
    r_lmn_opto, _ = pearsonr(popcoh['lmn']['wak'], popcoh['lmn']['opto'])

    results.append({
        'w_psb_lmn_': w_psb,
        'r_adn': r_adn,
        'r_lmn': r_lmn,
        'r_adn_opto': r_adn_opto,
        'r_lmn_opto': r_lmn_opto,
        'popcoh_adn_wak_mean': np.mean(popcoh['adn']['wak']),
        'popcoh_adn_sws_mean': np.mean(popcoh['adn']['sws']),
        'popcoh_lmn_wak_mean': np.mean(popcoh['lmn']['wak']),
        'popcoh_lmn_sws_mean': np.mean(popcoh['lmn']['sws']),
    })

# Convert to DataFrame
df = pd.DataFrame(results)
print(df)

# Plot lines
fig, axes = subplots(1, 2, figsize=(12, 5))

og_wak_sws = {"adn": 0.91, "lmn": 0.74}  # Original model values for reference
og_wak_opto = {"adn": 0.39, "lmn": 0.08}

for i, (col, col_opto, title) in enumerate(zip(['r_adn', 'r_lmn'], ['r_adn_opto', 'r_lmn_opto'], ['ADN', 'LMN'])):
    ax = axes[i]
    ax.plot(df['w_psb_lmn_'], df[col], 'o-', label='wake vs sws')
    ax.plot(df['w_psb_lmn_'], df[col_opto], 'o-', label='wake vs opto')
    ax.set_xlabel('w_psb_lmn_')
    ax.set_ylabel('correlation')
    ax.set_title(f'{title} popcoh correlation')
    ax.axhline(og_wak_sws[title.lower()], color='C0', linestyle='--')
    ax.axhline(og_wak_opto[title.lower()], color='C1', linestyle='--')
    ax.set_ylim(-0.5, 1.1)
    ax.legend()

tight_layout()
show()
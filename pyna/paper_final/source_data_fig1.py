# -*- coding: utf-8 -*-
# Export the Source Data of Figure 1 (population panels only, no example session) to an Excel file.
# Computations mirror main_pyna_fig1.py.
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import mannwhitneyu
import _pickle as cPickle
import os
import sys

from source_data_utils import write_figure_sheet

try:
    from functions import *
except:
    sys.path.append("../")
    from functions import *

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

###############################################################################################
# LOADING DATA
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
allrlmn = data['allr']
r_lmn = data['pearsonr']

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'rb'))
allradn = data['allr']
r_adn = data['pearsonr']

allr = {'adn': allradn, 'lmn': allrlmn}
allr_sess = {'adn': r_adn, 'lmn': r_lmn}

names = {'adn': "ADN", 'lmn': "LMN"}
sheets = {}
readme = []

###############################################################################################
# Fig 1I : pairwise correlations wake vs sleep
###############################################################################################
for g in ['adn', 'lmn']:
    tmp = allr[g].dropna()
    r, p = scipy.stats.pearsonr(tmp['wak'].values, tmp['sws'].values)
    df = pd.DataFrame({"neuron 1": [a for a, b in tmp.index], "neuron 2": [b for a, b in tmp.index],
                       "wake corr. (r)": tmp['wak'].values, "sleep corr. (r)": tmp['sws'].values})
    sheets[f"Fig1I_{names[g]}"] = df
    pstr = f"p = {p:.3g}" if p > 0 else "p < 1e-300 (below floating-point precision)"
    readme.append((f"Fig1I_{names[g]}",
                   f"{names[g]}: Pearson correlation of each neuron pair during wake vs non-REM sleep. n = {len(tmp)} pairs; "
                   f"Pearson r = {r:.4f}, {pstr}."))

###############################################################################################
# Fig 1J : population coherence per session
###############################################################################################
tmp = [allr_sess[g]['sws'] for g in ['adn', 'lmn']]
zw, p = mannwhitneyu(tmp[1], tmp[0])
rows = []
for g, t in zip(['adn', 'lmn'], tmp):
    rows.append(pd.DataFrame({"region": names[g], "session": t.index.values, "pop. coherence (r)": t.values}))
sheets["Fig1J"] = pd.concat(rows, ignore_index=True)
readme.append(("Fig1J",
               f"Population coherence (Pearson r between wake and non-REM sleep pairwise correlations) per session. "
               f"n = {len(tmp[0])} ADN sessions, n = {len(tmp[1])} LMN sessions. Horizontal bar / dot = mean "
               f"(ADN {tmp[0].mean():.4f}, LMN {tmp[1].mean():.4f}). Mann-Whitney U test (two-sided): U = {zw:.1f}, p = {p:.3g}."))

###############################################################################################
# Fig 1K : LMN-ADN cross-correlograms
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN-ADN.pickle'), 'rb'))
index = data['zorder'].index.values
zcc = data['zcc']
for k, label in zip(['wak', 'sws'], ['Wake', 'Sleep']):
    cc = zcc[k][index].loc[-0.02:0.02]
    summary = pd.DataFrame({"time lag (ms)": cc.index.values * 1000,
                            "mean (Z)": cc.mean(1).values, "SD (Z)": cc.std(1).values})
    # One row per pair, one column per time lag (keeps the sheet narrow)
    pairs = cc.T.copy()
    pairs.columns = [f"{lag:.1f} ms" for lag in cc.index.values * 1000]
    pairs.index = [f"{a} -> {b}" for a, b in cc.columns]
    pairs.index.name = "pair (LMN -> ADN)"
    sheets[f"Fig1K_{label}_mean"] = summary
    sheets[f"Fig1K_{label}_pairs"] = pairs.reset_index()
    readme.append((f"Fig1K_{label}_mean", f"{label}: z-scored LMN-ADN cross-correlograms, mean +/- SD across n = {len(index)} LMN-ADN pairs."))
    readme.append((f"Fig1K_{label}_pairs", f"{label}: z-scored cross-correlogram of each LMN-ADN pair (n = {len(index)} pairs; one row per pair LMN neuron -> ADN neuron, one column per time lag)."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("Fig1", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)

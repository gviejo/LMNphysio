# -*- coding: utf-8 -*-
# Export the Source Data of Supplementary Figure 3 (random sampling control) to Source_Data.xlsx.
# Computations mirror main_pyna_supp3.py.
import numpy as np
import pandas as pd
import scipy.stats
import _pickle as cPickle
import os

from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []


def fmt_p(p):
    return f"p = {p:.3g}" if p > 0 else "p < 1e-300"


data_rand = cPickle.load(open(os.path.join(dropbox_path, 'random_LMN_comparison.pickle'), 'rb'))
all_center_tc_real = data_rand['all_center_tc_real']
all_center_tc_fake = data_rand['all_center_tc_fake']
allr_fake = data_rand['allr']

data_real = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
allr_real = data_real['allr'].dropna()

###############################################################################################
# Supp Fig 3A : non-REM tuning curves, real vs sampled
###############################################################################################
n_bins = all_center_tc_real.shape[0]
df = {"angle from peak (deg)": np.linspace(-180, 180, n_bins)}
for label, tc in zip(["Real", "Sampled"], [all_center_tc_real, all_center_tc_fake]):
    df[f"{label} mean rate (Hz)"] = np.nanmean(tc, 1)
    df[f"{label} SEM"] = np.nanstd(tc, 1) / np.sqrt(np.sum(np.isfinite(tc), 1))
    df[f"{label} n neurons"] = np.sum(np.isfinite(tc), 1)
sheets["SuppFig3A"] = pd.DataFrame(df)
readme.append(("SuppFig3A",
               f"Centered non-REM tuning curves of LMN neurons, real vs sampled (Poisson) spike trains, "
               f"mean +/- SEM across n = {all_center_tc_real.shape[1]} neurons (n of non-NaN neurons per bin also given)."))

###############################################################################################
# Supp Fig 3B, C : pairwise correlations wake vs non-REM, sampled and real
###############################################################################################
for P, label, allr in zip(["B", "C"], ["Sampled", "Real"], [allr_fake, allr_real]):
    r, p = scipy.stats.pearsonr(allr['wak'].values, allr['sws'].values)
    m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
    sheets[f"SuppFig3{P}"] = pd.DataFrame({
        "neuron 1": [k[0] for k in allr.index.values],
        "neuron 2": [k[1] for k in allr.index.values],
        "Wake (r)": allr['wak'].values,
        "non-REM (r)": allr['sws'].values,
    })
    readme.append((f"SuppFig3{P}",
                   f"{label} LMN: correlation of each neuron pair during wake and non-REM sleep (n = {len(allr)} pairs). "
                   f"Pearson r = {r:.4f}, {fmt_p(p)} (two-sided). Linear fit: slope = {m:.4f}, intercept = {b:.4f}."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("SuppFig3", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

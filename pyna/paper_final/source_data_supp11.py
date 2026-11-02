# -*- coding: utf-8 -*-
# Export the Source Data of Supplementary Figure 11 (model) to Source_Data.xlsx.
# Computations mirror main_pyna_supp11.py. Panels D and E (activity of one simulation) are not exported.
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import _pickle as cPickle
import os

sys.path.append("../../model/")
from model import Model
from source_data_utils import write_figure_sheet

sheets = {}
readme = []

names = {'adn': "ADN", 'lmn': "LMN"}

# Same parameters as the simulations of Fig. 6 (model/main_model4_rings_seeds.py)
data_seeds = cPickle.load(open("/home/gviejo/Dropbox/LMNphysio/model/model_popcoh_seeds.pickle", 'rb'))
popcoh_seeds = data_seeds['popcoh']

m = Model(**data_seeds['conditions']['sws'])

###############################################################################################
# Supp Fig 11A, B : weight matrices
###############################################################################################
w = pd.DataFrame(m.w_lmn_adn, columns=[f"LMN {j}" for j in range(m.w_lmn_adn.shape[1])])
w.insert(0, "ADN neuron", np.arange(m.w_lmn_adn.shape[0]))
sheets["SuppFig11A"] = w
readme.append(("SuppFig11A",
               f"Model weight matrix W LMN->ADN (rows = {m.w_lmn_adn.shape[0]} ADN neurons, columns = {m.w_lmn_adn.shape[1]} LMN neurons)."))

# Transposed so that the sheet stays narrow: one row per PSB neuron.
w = pd.DataFrame(m.w_psb_lmn.T, columns=[f"LMN {j}" for j in range(m.w_psb_lmn.shape[0])])
w.insert(0, "PSB (ADN) neuron", np.arange(m.w_psb_lmn.shape[1]))
sheets["SuppFig11B"] = w
readme.append(("SuppFig11B",
               f"Model weight matrix W PSB->LMN, transposed: rows = {m.w_psb_lmn.shape[1]} PSB (ADN) neurons, "
               f"columns = {m.w_psb_lmn.shape[0]} LMN neurons (the figure shows LMN on the y-axis)."))

###############################################################################################
# Supp Fig 11C : pairwise correlations in the model
###############################################################################################
popcoh = cPickle.load(open("/home/gviejo/Dropbox/LMNphysio/model/model_rings.pickle", 'rb'))['popcoh']

summary = []
for st in ['adn', 'lmn']:
    df = popcoh[st][['wak', 'sws', 'opto']].copy()
    df.columns = ["'Wake' corr. (r)", "'Sleep' corr. (r)", "'Opto.' corr. (r)"]
    df.insert(0, "neuron pair", np.arange(len(df)) + 1)
    sheets[f"SuppFig11C_pairs_{names[st]}"] = df
    readme.append((f"SuppFig11C_pairs_{names[st]}",
                   f"Model {names[st]}, example simulation of Fig. 6 (seed 0): pairwise correlation of each simulated neuron pair "
                   f"(n = {len(df)} pairs) in 'Wake', 'Sleep' and 'Opto.' "
                   f"(the figure shows the occupied bins of the 2D histogram, 99 x 99 bins between -1 and 1)."))
    for k, cond in zip(['sws', 'opto'], ["'Sleep'", "'Opto.'"]):
        r, p = pearsonr(popcoh[st]['wak'], popcoh[st][k])
        a, b = np.polyfit(popcoh[st]['wak'], popcoh[st][k], 1)
        r_seeds = popcoh_seeds[f"{st}_{k}"]
        summary.append({"region": names[st], "condition": cond, "n pairs": len(df),
                        "example r (seed 0)": r, "example p (two-sided)": p, "example fit slope": a, "example fit intercept": b,
                        "n simulations": len(r_seeds), "mean r across simulations": r_seeds.mean(), "SD r across simulations": r_seeds.std()})

sheets["SuppFig11C"] = pd.DataFrame(summary)
readme.append(("SuppFig11C",
               "Pearson r between 'Wake' and 'Sleep' or 'Opto.' pairwise correlations. Example simulation (seed 0): r, p and linear fit (line). "
               "Value shown in each panel: mean +/- SD of r across simulations with different random seeds (per-seed values in Fig6 sheet, block Fig6D model seeds)."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("SuppFig11", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)
print(sheets["SuppFig11C"])

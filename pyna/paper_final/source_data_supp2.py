# -*- coding: utf-8 -*-
# Export the Source Data of Supplementary Figure 2 (population panels only, no example session) to Source_Data.xlsx.
# Computations mirror main_pyna_supp2.py. Panel B (REM example session) is not exported.
# The random-sampling panels at the bottom of the script are shown in Supplementary Figure 3 (source_data_supp3.py).
import sys
import numpy as np
import pandas as pd
import scipy.stats
import _pickle as cPickle
import os

sys.path.append("../")
from functions import centerTuningCurves_with_peak
from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []

names = {'adn': "ADN", 'lmn': "LMN"}


def fmt_p(p):
    return f"p = {p:.3g}" if p > 0 else "p < 1e-300"


###############################################################################################
# Supp Fig 2A : HD cell selection and tuning curves
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'All_TC.pickle'), 'rb'))
alltc = data['alltc']
allinfo = data['allinfo']

for st in ['adn', 'lmn']:
    idx = allinfo[allinfo['location'] == st].index.values
    info = allinfo.loc[idx]
    tokeep = info[(info['rate'] > 1.0) & (info['SI'] > 0.1) & (info['tokeep'] == 1)].index.values

    sheets[f"SuppFig2A_selection_{names[st]}"] = pd.DataFrame({
        "neuron": idx,
        "rate (Hz)": info['rate'].values,
        "HD info. (bits/spk)": info['SI'].values,
        "selected HD neuron": np.isin(idx, tokeep).astype(int),
    })
    readme.append((f"SuppFig2A_selection_{names[st]}",
                   f"{names[st]}: mean firing rate vs HD information of each neuron (n = {len(idx)} neurons). "
                   f"Selected HD neurons (rate > 1 Hz, HD info. > 0.1 bits/spk): n = {len(tokeep)} (colored dots)."))

    tc = centerTuningCurves_with_peak(alltc[tokeep])
    tc = tc / tc.max()
    sheets[f"SuppFig2A_tuning_mean_{names[st]}"] = pd.DataFrame({
        "angle (deg)": np.rad2deg(tc.index.values),
        "mean norm. rate": tc.mean(1).values,
        "SD": tc.std(1).values,
    })
    readme.append((f"SuppFig2A_tuning_mean_{names[st]}",
                   f"{names[st]}: centered tuning curves normalized to their maximum, mean +/- SD across n = {tc.shape[1]} HD neurons."))

    heat = tc.T.copy()
    heat.columns = [f"{a:.1f} deg" for a in np.rad2deg(tc.index.values)]
    heat.insert(0, "neuron", tc.columns.values)
    sheets[f"SuppFig2A_tuning_heatmap_{names[st]}"] = heat
    readme.append((f"SuppFig2A_tuning_heatmap_{names[st]}",
                   f"{names[st]}: centered normalized tuning curve of each HD neuron (n = {tc.shape[1]}; one row per neuron, color scale = rate)."))

###############################################################################################
# Supp Fig 2C : population coherence wake vs REM
###############################################################################################
allr_sess = {st: cPickle.load(open(os.path.join(dropbox_path, f'All_correlation_{names[st]}.pickle'), 'rb'))['pearsonr']
             for st in ['adn', 'lmn']}

rows = []
for st in ['adn', 'lmn']:
    tmp = allr_sess[st]['rem']
    rows.append(pd.DataFrame({"region": names[st], "session": tmp.index.values,
                              "pop. coherence wake vs REM (r)": tmp.values.astype(float)}))
sheets["SuppFig2C"] = pd.concat(rows, ignore_index=True)

tmp = [allr_sess[st]['rem'].astype(float) for st in ['adn', 'lmn']]
zw, p = scipy.stats.mannwhitneyu(tmp[1], tmp[0])
readme.append(("SuppFig2C",
               f"Population coherence between wake and REM sleep per session (Pearson r of pairwise correlations). "
               f"n = {len(tmp[0])} ADN sessions, n = {len(tmp[1])} LMN sessions. Line / dot = mean "
               f"(ADN {tmp[0].mean():.4f}, LMN {tmp[1].mean():.4f}). The bottom plot shows the same values as a distribution. "
               f"Mann-Whitney U test LMN vs ADN (two-sided): U = {zw:.1f}, {fmt_p(p)}."))

###############################################################################################
# Supp Fig 2D, E : pairwise correlations per animal
###############################################################################################
for P, st in zip(["D", "E"], ['adn', 'lmn']):
    data = cPickle.load(open(os.path.join(dropbox_path, f'All_correlation_{names[st]}.pickle'), 'rb'))
    allr = data['allr']
    sessions = [s.split("/")[-1] for s in data['pearsonr'].index.values]
    animals = np.unique([s.split("-")[0] for s in sessions])

    rows = []
    summary = []
    for a in animals:
        index = [k for k in allr.index.values if a in k[0]]
        tmp = allr.loc[index]
        rows.append(pd.DataFrame({"animal": a,
                                  "neuron 1": [k[0] for k in index],
                                  "neuron 2": [k[1] for k in index],
                                  "Wake (r)": tmp['wak'].values,
                                  "non-REM (r)": tmp['sws'].values,
                                  "REM (r)": tmp['rem'].values}))
        row = {"animal": a, "n pairs": len(index)}
        for e, label in zip(['sws', 'rem'], ["non-REM", "REM"]):
            r, pr = scipy.stats.pearsonr(tmp['wak'].values, tmp[e].values)
            m, b = np.polyfit(tmp['wak'].values, tmp[e].values, 1)
            row[f"Wake vs {label} r"] = r
            row[f"Wake vs {label} p"] = pr
            row[f"{label} fit slope"] = m
            row[f"{label} fit intercept"] = b
        summary.append(row)

    sheets[f"SuppFig2{P}_pairs"] = pd.concat(rows, ignore_index=True)
    readme.append((f"SuppFig2{P}_pairs",
                   f"{names[st]}: correlation of each neuron pair during wake, non-REM and REM sleep, grouped by animal "
                   f"(n = {len(animals)} animals, n = {len(allr)} pairs)."))
    sheets[f"SuppFig2{P}_per_animal"] = pd.DataFrame(summary)
    readme.append((f"SuppFig2{P}_per_animal",
                   f"{names[st]}: per animal, number of pairs, Pearson r (and two-sided p) between wake and non-REM / REM pairwise "
                   f"correlations (value shown in each panel), and linear fit (black line)."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("SuppFig2", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)
for P in ["D", "E"]:
    print(sheets[f"SuppFig2{P}_per_animal"][["animal", "n pairs", "Wake vs non-REM r", "Wake vs REM r"]].round(2))

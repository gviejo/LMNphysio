# -*- coding: utf-8 -*-
# Export the Source Data of Supplementary Figure 7 (PSB stimulation in PSB) to Source_Data.xlsx.
# Computations mirror main_pyna_supp7.py. Panel A is histology and panel B shows an example session: not exported.
import sys
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import _pickle as cPickle
import os

sys.path.append("../")
from functions import centerTuningCurves_with_peak
from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []


def fmt_p(p):
    return f"p = {p:.3g}" if p > 0 else "p < 1e-300"


psbdata = cPickle.load(open(dropbox_path + "/OPTO_PSB.pickle", "rb"))

groups = {}
allSI = []
allTC = []
waveforms = []
for gr in ["opto", "ctrl"]:
    groups[gr] = {}
    for ep in ['wake', 'sleep']:
        allmeta = psbdata['allmeta'][gr][ep]
        groups[gr][ep] = allmeta.index.values
        allTC.append(psbdata['alltc'][gr][ep])
        allSI.append(allmeta['SI'])
        waveforms.append(psbdata['allmwf'][gr][ep])

allSI = pd.concat(allSI, axis=0)
allTC = pd.concat(allTC, axis=1)
waveforms = pd.concat(waveforms, axis=1)
mwf = waveforms.apply(lambda col: zscore(col.values), axis=0)

through_to_peak = (mwf.idxmax() - mwf.idxmin())
through_to_peak = through_to_peak[through_to_peak > 0.0]

tokeep = np.intersect1d(through_to_peak.index, allSI.index)
hd_idx = tokeep[(through_to_peak[tokeep] > 0.0004) & (allSI[tokeep] > 0.5)]
nhd_idx = tokeep[(through_to_peak[tokeep] < 0.0004) & (allSI[tokeep] < 0.5)]
unk_idx = np.setdiff1d(tokeep, np.union1d(hd_idx, nhd_idx))

cell_type = pd.Series("Unknown", index=tokeep)
cell_type[hd_idx] = "HD"
cell_type[nhd_idx] = "FS"

###############################################################################################
# Supp Fig 7C : cell classification and waveforms
###############################################################################################
sheets["SuppFig7C_classification"] = pd.DataFrame({
    "neuron": tokeep,
    "trough-to-peak (ms)": through_to_peak[tokeep].values * 1e3,
    "SI (bits/spk)": allSI[tokeep].values,
    "cell type": cell_type.values,
})
readme.append(("SuppFig7C_classification",
               f"PSB neurons recorded during PSB stimulation sessions (Chrimson and TdTomato): trough-to-peak duration vs "
               f"HD information (SI). HD: trough-to-peak > 0.4 ms and SI > 0.5 (n = {len(hd_idx)}); "
               f"FS: trough-to-peak < 0.4 ms and SI < 0.5 (n = {len(nhd_idx)}); Unknown: n = {len(unk_idx)}."))

sheets["SuppFig7C_waveforms"] = pd.DataFrame({
    "time (ms)": mwf.index.values * 1e3,
    "HD mean (z)": mwf[hd_idx].mean(1).values,
    "HD SD": mwf[hd_idx].std(1).values,
    "FS mean (z)": mwf[nhd_idx].mean(1).values,
    "FS SD": mwf[nhd_idx].std(1).values,
})
readme.append(("SuppFig7C_waveforms",
               f"Mean waveform (z-scored per neuron), mean +/- SD across n = {len(hd_idx)} HD and n = {len(nhd_idx)} FS neurons. "
               f"Note: the time axis is in ms (the figure axis is labelled 0.000-0.001 with unit ms)."))

###############################################################################################
# Supp Fig 7D : tuning curves and firing rate during stimulation
###############################################################################################
labels = {"hd": "HD", "fs": "FS"}
for grn, idx in zip(['hd', 'fs'], [hd_idx, nhd_idx]):
    tc = centerTuningCurves_with_peak(allTC[idx])
    tc = tc / tc.loc[0]
    out = tc.T.copy()
    out.columns = [f"{a:.1f} deg" for a in np.rad2deg(tc.index.values)]
    out.insert(0, "neuron", tc.columns.values)
    out.loc["mean"] = ["mean"] + list(tc.mean(1).values)
    sheets[f"SuppFig7D_tuning_{labels[grn]}"] = out
    readme.append((f"SuppFig7D_tuning_{labels[grn]}",
                   f"PoSub {labels[grn]}: centered tuning curve of each neuron normalized to its value at the peak "
                   f"(n = {tc.shape[1]}; one row per neuron, last row = mean)."))

gr = "opto"
for ep, sl, stim in zip(['wake', 'sleep'], [slice(-4, 14), slice(-1, 2)], ["0 to 10 s", "0 to 1 s"]):
    allfr = psbdata['allfr'][gr][ep]
    for grn, idx in zip(['hd', 'fs'], [hd_idx, nhd_idx]):
        idx = np.intersect1d(idx, groups[gr][ep])
        tmp = allfr[idx].loc[sl]
        tmp = tmp.apply(lambda col: gaussian_filter1d(col.values, sigma=1), axis=0)
        out = tmp.T.copy()
        out.columns = [f"{t:.2f} s" for t in tmp.index.values]
        out.insert(0, "neuron", tmp.columns.values)
        out.loc["mean"] = ["mean"] + list(tmp.mean(1).values)
        sheets[f"SuppFig7D_rate_{labels[grn]}_{ep}"] = out
        readme.append((f"SuppFig7D_rate_{labels[grn]}_{ep}",
                       f"Chrimson, {'wake' if ep == 'wake' else 'non-REM sleep'}: normalized firing rate of each PoSub {labels[grn]} neuron "
                       f"around PSB stimulation (stimulation {stim}), Gaussian smoothing sigma = 1 bin "
                       f"(n = {len(idx)}; one row per neuron, last row = mean)."))

###############################################################################################
# Supp Fig 7E : firing rate modulation, Chrimson vs TdTomato
###############################################################################################
for grn, idx in zip(['hd', 'fs'], [hd_idx, nhd_idx]):
    rows = []
    stats = []
    for ep, sl, msl in zip(['wake', 'sleep'], [slice(1, 10), slice(0.1, 1)], [slice(-5, -1), slice(-1, -0.1)]):
        ys = []
        for gr, glabel in zip(['opto', 'ctrl'], ["Chrimson", "TdTomato"]):
            tmp = psbdata['allfr'][gr][ep][np.intersect1d(idx, groups[gr][ep])]
            y = (tmp.loc[sl].mean().values - tmp.loc[msl].mean().values) / (tmp.loc[sl].mean().values + tmp.loc[msl].mean().values)
            rows.append(pd.DataFrame({"epoch": "Wake" if ep == 'wake' else "non-REM", "group": glabel,
                                      "neuron": tmp.columns.values, "FR mod.": y}))
            ys.append(y)
        zw, p = scipy.stats.mannwhitneyu(ys[0], ys[1])
        stats.append(f"{'Wake' if ep == 'wake' else 'non-REM'}: n = {len(ys[0])} Chrimson, n = {len(ys[1])} TdTomato neurons, "
                     f"mean {ys[0].mean():.4f} vs {ys[1].mean():.4f}, U = {zw:.1f}, {fmt_p(p)}")
    sheets[f"SuppFig7E_{labels[grn]}"] = pd.concat(rows, ignore_index=True)
    readme.append((f"SuppFig7E_{labels[grn]}",
                   f"PoSub {labels[grn]}: firing rate modulation of each neuron, (stim - pre) / (stim + pre), "
                   f"stim = 1-10 s (wake) or 0.1-1 s (non-REM), pre = -5 to -1 s (wake) or -1 to -0.1 s (non-REM). "
                   f"Line = mean. Mann-Whitney U test Chrimson vs TdTomato (two-sided): " + "; ".join(stats) + "."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("SuppFig7", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

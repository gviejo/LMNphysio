# -*- coding: utf-8 -*-
# Export the Source Data of Figure 2 (population panels only, no example session) to Source_Data.xlsx.
# Computations mirror main_pyna_fig2.py.
import numpy as np
import pandas as pd
import scipy.stats
from scipy.ndimage import gaussian_filter1d
import _pickle as cPickle
import os
import sys

from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []

def pair_names(columns):
    return [f"{a} -> {b}" for a, b in columns]

###############################################################################################
# Fig 2F : PSB -> LMN connections
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN-PSB.pickle'), 'rb'))
pospeak = data['pospeak']
negpeak = data['negpeak']
zcc = data['zcc']
order = data['order']

peaks = pd.concat([pospeak, negpeak])
sheets["Fig2F_peak_lags"] = pd.DataFrame({
    "pair (PSB -> LMN)": pair_names(peaks.index),
    "peak lag (ms)": peaks.values * 1000,
    "peak sign": ["positive lag"] * len(pospeak) + ["negative lag"] * len(negpeak),
})
readme.append(("Fig2F_peak_lags",
               f"Lag of the cross-correlogram peak for PSB-LMN pairs with Z > 3: n = {len(pospeak)} pairs with positive lag, "
               f"n = {len(negpeak)} pairs with negative lag (n = {len(peaks)} pairs in total)."))

bin_edges = np.linspace(-0.01, 0.01, 50)
hist = {}
for label, peak in zip(["positive lag", "negative lag"], [pospeak, negpeak]):
    hist[label], _ = np.histogram(peak.values, bins=bin_edges, range=(-0.01, 0.01))
df = pd.DataFrame({"bin start (ms)": bin_edges[:-1] * 1000, "bin end (ms)": bin_edges[1:] * 1000,
                   "count, positive-lag pairs": hist["positive lag"], "count, negative-lag pairs": hist["negative lag"]})
sheets["Fig2F_histogram"] = df
readme.append(("Fig2F_histogram", "Histogram of peak lags (top plot): number of pairs per lag bin."))

zs = zcc.apply(lambda x: gaussian_filter1d(x, sigma=1))
a = zs[order].loc[0.002:0.008].idxmax().sort_values().index
Z = zs[a].loc[-0.02:0.02]
heat = Z.T.copy()
heat.columns = [f"{lag:.1f} ms" for lag in Z.index.values * 1000]
heat.insert(0, "pair (PSB -> LMN)", pair_names(Z.columns))
sheets["Fig2F_heatmap"] = heat
readme.append(("Fig2F_heatmap",
               f"Z-scored cross-correlograms (Gaussian smoothing sigma = 1 bin) of the n = {len(order)} PSB-LMN pairs with a positive-lag peak, "
               f"sorted by peak lag (one row per pair, bottom to top in the figure; one column per time lag)."))

###############################################################################################
# Fig 2G : PSB-LMN normalized cross-correlograms
###############################################################################################
def get_zscore(cc, w=0.02):
    # Smoothing with big gaussian
    sigma = int(w / np.median(np.diff(cc.index.values))) + 1
    df = cc.apply(lambda col: gaussian_filter1d(col, sigma=sigma))

    # Zscoring
    tmp = cc - df
    zcc = tmp / tmp.std(0)
    return zcc

index = data['pospeak'].index.values
zcc = {"wak": get_zscore(data['allcc']['wak']), "sws": get_zscore(data['allcc']['sws'])}

for k, label in zip(['wak', 'sws'], ['Wake', 'Sleep']):
    cc = zcc[k][index].loc[-0.02:0.02]
    sheets[f"Fig2G_{label}_mean"] = pd.DataFrame({"time lag (ms)": cc.index.values * 1000,
                                                  "mean (Z)": cc.mean(1).values, "SD (Z)": cc.std(1).values})
    pairs = cc.T.copy()
    pairs.columns = [f"{lag:.1f} ms" for lag in cc.index.values * 1000]
    pairs.insert(0, "pair (PSB -> LMN)", pair_names(cc.columns))
    sheets[f"Fig2G_{label}_pairs"] = pairs
    readme.append((f"Fig2G_{label}_mean", f"{label}: normalized PSB-LMN cross-correlograms, mean +/- SD across n = {len(index)} pairs."))
    readme.append((f"Fig2G_{label}_pairs", f"{label}: normalized cross-correlogram of each PSB-LMN pair (n = {len(index)} pairs; one row per pair, one column per time lag)."))

###############################################################################################
# Fig 2J : LMN population coherence during Down and Up states
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'CORR_LMN-PSB_UP_DOWN.pickle'), 'rb'))
pearson = data['pearson']
baseline = data['baseline']

df = pd.DataFrame({
    "session": pearson.index.values,
    "pop. coherence Down (r)": pearson['down'].values.astype(float),
    "pop. coherence Up (r)": pearson['decimated'].values.astype(float),
    "baseline Down (r)": baseline['down'].values.astype(float),
    "baseline Up (r)": baseline['decimated'].values.astype(float),
})
df["Down minus baseline"] = df["pop. coherence Down (r)"] - df["baseline Down (r)"]
df["Up minus baseline"] = df["pop. coherence Up (r)"] - df["baseline Up (r)"]
sheets["Fig2J"] = df

stats = []
for k, label in zip(['down', 'decimated'], ['Down', 'Up']):
    zw, p = scipy.stats.wilcoxon(pearson[k].values.astype("float"), baseline[k].values.astype("float"), alternative='greater')
    stats.append(f"{label} vs baseline: W = {zw:.1f}, p = {p:.4g}")
tmp = (pearson[['down', 'decimated']] - baseline[['down', 'decimated']]).values.astype("float")
zw, p = scipy.stats.wilcoxon(tmp[:, 1], tmp[:, 0], alternative='greater')
stats.append(f"Up vs Down (minus baseline): W = {zw:.1f}, p = {p:.4g}")
readme.append(("Fig2J",
               f"LMN population coherence per session during PSB Down and Up states, and the same minus baseline. n = {len(pearson)} sessions. "
               f"Dot in the bottom plot = mean (Down {pearson['down'].mean():.4f}, Up {pearson['decimated'].mean():.4f}). "
               f"Wilcoxon signed-rank tests (one-sided, greater): " + "; ".join(stats) + "."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("Fig2", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)

# -*- coding: utf-8 -*-
# Export the Source Data of Figure 4 (population panels only, no example session) to Source_Data.xlsx.
# Computations mirror main_pyna_fig4.py.
import numpy as np
import pandas as pd
import scipy.stats
from scipy.ndimage import gaussian_filter1d
import _pickle as cPickle
import os

from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []

sides = {"ipsi": "Ipsilateral", "bilateral": "Bilateral"}


def fmt_p(p):
    return f"p = {p:.3g}" if p > 0 else "p < 1e-300"


for ranges, name, P_rate, P_corr, T in zip(
        [(-7, 0, 10, 17), (-0.7, 0, 1, 1.7)],
        ["WAKE", "SLEEP"],
        ["G", "I"],
        ["H", "J"],
        ["Wakefulness", "Non-REM sleep"]):

    data = cPickle.load(open(os.path.join(dropbox_path, f"OPTO_{name}.pickle"), 'rb'))
    corr = data['corr']
    change_fr = data['change_fr']
    allfr = data['allfr']
    baseline = data['baseline']

    ###########################################################################################
    # Fig 4G, I : ADN firing rate during ipsilateral / bilateral PSB stimulation
    ###########################################################################################
    for f in ['ipsi', 'bilateral']:
        tmp = allfr['adn']['opto'][f]
        tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1.5, mode='constant'))
        tmp = tmp.loc[ranges[0]:ranges[-1]]
        sheets[f"Fig4{P_rate}_rate_mean_{f}"] = pd.DataFrame({"opto. time (s)": tmp.index.values,
                                                              "mean norm. rate (a.u.)": tmp.mean(1).values,
                                                              "SD": tmp.std(1).values})
        rates = tmp.T.copy()
        rates.columns = [f"{t:.2f} s" for t in tmp.index.values]
        rates.insert(0, "neuron", tmp.columns.values)
        sheets[f"Fig4{P_rate}_rate_neurons_{f}"] = rates
        readme.append((f"Fig4{P_rate}_rate_mean_{f}",
                       f"{T}, {sides[f]} stimulation: normalized firing rate of ADN neurons around PSB stimulation "
                       f"(stimulation from {ranges[1]} to {ranges[2]} s), mean +/- SD across n = {tmp.shape[1]} neurons "
                       f"(Gaussian smoothing sigma = 1.5 bins)."))
        readme.append((f"Fig4{P_rate}_rate_neurons_{f}",
                       f"{T}, {sides[f]} stimulation: normalized firing rate of each ADN neuron (n = {tmp.shape[1]}; one row per neuron, one column per time bin)."))

        chfr = change_fr['adn']['opto'][f]
        delta = (chfr['opto'] - chfr['pre']) / chfr['pre']
        zw, p = scipy.stats.wilcoxon(np.array(delta.values).astype(float))
        sheets[f"Fig4{P_rate}_rate_mod_{f}"] = pd.DataFrame({"neuron": delta.index.values,
                                                             "rate pre": chfr['pre'].values,
                                                             "rate opto": chfr['opto'].values,
                                                             "rate mod. (opto-pre)/pre": delta.values})
        readme.append((f"Fig4{P_rate}_rate_mod_{f}",
                       f"{T}, {sides[f]} stimulation: rate modulation of each ADN neuron, (opto - pre) / pre. n = {len(delta)} neurons, "
                       f"mean = {np.mean(delta.values.astype(float)):.4f}. "
                       f"Wilcoxon signed-rank test against 0 (two-sided): W = {zw:.1f}, {fmt_p(p)}."))

    bins = np.linspace(-1, 1, 31)
    hist = {"bin start": bins[:-1], "bin end": bins[1:]}
    for f in ['ipsi', 'bilateral']:
        chfr = change_fr['adn']['opto'][f]
        delta = (chfr['opto'] - chfr['pre']) / chfr['pre']
        hist[f"proportion {sides[f]} (%)"] = 100 * np.histogram(delta.values, bins=bins)[0] / len(delta.values)
    sheets[f"Fig4{P_rate}_rate_mod_hist"] = pd.DataFrame(hist)
    readme.append((f"Fig4{P_rate}_rate_mod_hist",
                   f"{T}: histogram of rate modulation, percentage of neurons per bin (values outside [-1, 1] are not counted)."))

    ###########################################################################################
    # Fig 4H, J : ADN population coherence, Chrimson vs matched-rate control
    ###########################################################################################
    for f in ['ipsi', 'bilateral']:
        corr3 = corr['adn']['opto'][f]
        corr3 = corr3[corr3['n'] > 4]
        corr3 = corr3[~corr3.index.duplicated(keep='first')]
        base3 = baseline['adn']['opto'][f].loc[corr3.index.values]

        corr3 = corr3[['opto', 'decimated']].astype(float)
        base3 = base3[['opto', 'decimated']].astype(float)
        tmp = corr3 - base3

        sheets[f"Fig4{P_corr}_{f}"] = pd.DataFrame({
            "session": corr3.index.values,
            "pop. coherence Chrimson (r)": corr3['opto'].values,
            "pop. coherence Control (r)": corr3['decimated'].values,
            "baseline Chrimson (r)": base3['opto'].values,
            "baseline Control (r)": base3['decimated'].values,
            "Chrimson minus baseline": tmp['opto'].values,
            "Control minus baseline": tmp['decimated'].values,
        })

        stats = []
        for i, label in enumerate(['Chrimson', 'Control']):
            zw, p = scipy.stats.mannwhitneyu(corr3.iloc[:, i], base3.iloc[:, i], alternative='greater')
            stats.append(f"{label} vs baseline (one-sided, greater): U = {zw:.1f}, {fmt_p(p)}")
        zw, p = scipy.stats.mannwhitneyu(tmp['opto'].dropna(), tmp['decimated'].dropna())
        stats.append(f"Chrimson vs Control, minus baseline (two-sided): U = {zw:.1f}, {fmt_p(p)}")
        m = tmp.mean(0).values
        readme.append((f"Fig4{P_corr}_{f}",
                       f"{T}, {sides[f]} stimulation: ADN population coherence during PSB stimulation (Chrimson) vs control epochs "
                       f"with matched firing rate, per session (sessions with > 4 ADN neurons). n = {len(corr3)} sessions. "
                       f"Dot in the bottom plot = mean minus baseline (Chrimson {m[0]:.4f}, Control {m[1]:.4f}). "
                       f"Mann-Whitney U tests: " + "; ".join(stats) + "."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("Fig4", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

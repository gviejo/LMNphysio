# -*- coding: utf-8 -*-
# Export the Source Data of Figure 3 (population panels only, no example session) to Source_Data.xlsx.
# Computations mirror main_pyna_fig3.py.
import numpy as np
import pandas as pd
import scipy.stats
from scipy.ndimage import gaussian_filter1d
import _pickle as cPickle
import os
import sys

try:
    from functions import *
except:
    sys.path.append("../")
    from functions import *

from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []

ranges = {
    "OPTO_SLEEP": (-0.9, 0, 1, 1.9),
    "OPTO_WAKE": (-9, 0, 10, 19)
}
panels = {"OPTO_WAKE": {"rate": "E", "corr": "F"},
          "OPTO_SLEEP": {"rate": "I", "corr": "J"}}
titles = {"OPTO_WAKE": "Wakefulness", "OPTO_SLEEP": "Non-REM sleep"}
groups = {"opto": "Chrimson", "ctrl": "TdTomato"}


def get_corr(corr, baseline, keys):
    # Sessions with more than 4 neurons, as in main_pyna_fig3.py
    st, gr, sd, k = keys
    corr2 = corr[st][gr][sd]
    corr2 = corr2[corr2['n'] > 4][k]
    idx = corr2.index.values
    return idx, corr2.values.astype(float), baseline[st][gr][sd][k].loc[idx].values.astype(float)


def fmt_p(p):
    return f"p = {p:.3g}" if p > 0 else "p < 1e-300"


for f in ['OPTO_WAKE', 'OPTO_SLEEP']:
    data = cPickle.load(open(os.path.join(dropbox_path, f"{f}.pickle"), 'rb'))
    corr = data['corr']
    # OPTO_WAKE.pickle stores each session twice in corr (fixed in OPTO/main_pyna_OPTO_WAKE.py): keep one row per session
    for st in corr:
        for gr in corr[st]:
            for sd in corr[st][gr]:
                corr[st][gr][sd] = corr[st][gr][sd][~corr[st][gr][sd].index.duplicated(keep='first')]
    change_fr = data['change_fr']
    allfr = data['allfr']
    baseline = data['baseline']
    P = panels[f]
    T = titles[f]

    ###########################################################################################
    # Fig 3E, I : firing rate during opto
    ###########################################################################################
    tmp = allfr['lmn']['opto']['ipsi']
    tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1.5, mode='constant'))
    tmp = tmp.loc[ranges[f][0]:ranges[f][-1]]
    sheets[f"Fig3{P['rate']}_rate_mean"] = pd.DataFrame({"opto. time (s)": tmp.index.values,
                                                         "mean norm. rate (a.u.)": tmp.mean(1).values,
                                                         "SD": tmp.std(1).values})
    rates = tmp.T.copy()
    rates.columns = [f"{t:.2f} s" for t in tmp.index.values]
    rates.insert(0, "neuron", tmp.columns.values)
    sheets[f"Fig3{P['rate']}_rate_neurons"] = rates
    readme.append((f"Fig3{P['rate']}_rate_mean",
                   f"{T}: normalized firing rate of LMN neurons around PSB stimulation (stimulation from {ranges[f][1]} to {ranges[f][2]} s), "
                   f"mean +/- SD across n = {tmp.shape[1]} neurons (Gaussian smoothing sigma = 1.5 bins)."))
    readme.append((f"Fig3{P['rate']}_rate_neurons",
                   f"{T}: normalized firing rate of each LMN neuron (n = {tmp.shape[1]}; one row per neuron, one column per time bin)."))

    chfr = change_fr['lmn']['opto']['ipsi']
    delta = (chfr['opto'] - chfr['pre']) / chfr['pre']
    zw, p = scipy.stats.wilcoxon(delta.values.astype(float))
    sheets[f"Fig3{P['rate']}_rate_mod"] = pd.DataFrame({"neuron": delta.index.values,
                                                        "rate pre": chfr['pre'].values,
                                                        "rate opto": chfr['opto'].values,
                                                        "rate mod. (opto-pre)/pre": delta.values})
    bins = np.linspace(-1, 1, 31)
    counts, _ = np.histogram(delta.values, bins=bins)
    sheets[f"Fig3{P['rate']}_rate_mod_hist"] = pd.DataFrame({"bin start": bins[:-1], "bin end": bins[1:], "count (neurons)": counts})
    readme.append((f"Fig3{P['rate']}_rate_mod",
                   f"{T}: rate modulation of each LMN neuron, (opto - pre) / pre. n = {len(delta)} neurons. "
                   f"Wilcoxon signed-rank test against 0 (two-sided): W = {zw:.1f}, {fmt_p(p)}."))
    readme.append((f"Fig3{P['rate']}_rate_mod_hist", f"{T}: histogram of rate modulation (number of neurons per bin; values outside [-1, 1] are not counted)."))

    ###########################################################################################
    # Fig 3F, J : population coherence Chrimson vs TdTomato
    ###########################################################################################
    corr3, base3, rows = [], [], []
    for gr in ['opto', 'ctrl']:
        idx, c, b = get_corr(corr, baseline, ('lmn', gr, 'ipsi', 'opto'))
        corr3.append(c)
        base3.append(b)
        rows.append(pd.DataFrame({"group": groups[gr], "session": idx, "pop. coherence (r)": c,
                                  "baseline (r)": b, "minus baseline": c - b}))
    sheets[f"Fig3{P['corr']}"] = pd.concat(rows, ignore_index=True)

    stats = []
    for j, gr in enumerate(['opto', 'ctrl']):
        zw, p = scipy.stats.mannwhitneyu(corr3[j], base3[j], alternative='greater')
        stats.append(f"{groups[gr]} vs baseline (one-sided, greater): U = {zw:.1f}, {fmt_p(p)}")
    tmp = [c - b for c, b in zip(corr3, base3)]
    zw, p = scipy.stats.mannwhitneyu(tmp[0], tmp[1])
    stats.append(f"Chrimson vs TdTomato, minus baseline (two-sided): U = {zw:.1f}, {fmt_p(p)}")
    readme.append((f"Fig3{P['corr']}",
                   f"{T}: LMN population coherence during PSB stimulation per session (sessions with > 4 LMN neurons). "
                   f"n = {len(corr3[0])} Chrimson sessions, n = {len(corr3[1])} TdTomato sessions. "
                   f"Dot in the bottom plot = mean (Chrimson {corr3[0].mean():.4f}, TdTomato {corr3[1].mean():.4f}). "
                   f"Mann-Whitney U tests: " + "; ".join(stats) + "."))

    ###########################################################################################
    # Fig 3K : Chrimson vs control with matched firing rate (sleep)
    ###########################################################################################
    if f == 'OPTO_SLEEP':
        idx, c_opto, b_opto = get_corr(corr, baseline, ('lmn', 'opto', 'ipsi', 'opto'))
        _, c_dec, b_dec = get_corr(corr, baseline, ('lmn', 'opto', 'ipsi', 'decimated'))
        corr3 = pd.DataFrame({'opto': c_opto, 'decimated': c_dec})
        base3 = pd.DataFrame({'opto': b_opto, 'decimated': b_dec})
        tmp = corr3 - base3
        sheets["Fig3K"] = pd.DataFrame({"session": idx,
                                        "pop. coherence Chrimson (r)": c_opto, "pop. coherence Control (r)": c_dec,
                                        "baseline Chrimson (r)": b_opto, "baseline Control (r)": b_dec,
                                        "Chrimson minus baseline": tmp['opto'].values,
                                        "Control minus baseline": tmp['decimated'].values})
        stats = []
        for j, (k, label) in enumerate(zip(['opto', 'decimated'], ['Chrimson', 'Control'])):
            zw, p = scipy.stats.mannwhitneyu(corr3.iloc[:, j], base3.iloc[:, j], alternative='greater')
            stats.append(f"{label} vs baseline (one-sided, greater): U = {zw:.1f}, {fmt_p(p)}")
        zw, p = scipy.stats.mannwhitneyu(tmp['opto'], tmp['decimated'])
        stats.append(f"Chrimson vs Control, minus baseline (two-sided): U = {zw:.1f}, {fmt_p(p)}")
        m = tmp.mean(0).values
        readme.append(("Fig3K",
                       f"Non-REM sleep: LMN population coherence during PSB stimulation (Chrimson) vs control epochs outside stimulation "
                       f"with matched firing rate, per session. n = {len(corr3)} sessions. "
                       f"Dot in the bottom plot = mean minus baseline (Chrimson {m[0]:.4f}, Control {m[1]:.4f}). "
                       f"Mann-Whitney U tests: " + "; ".join(stats) + "."))

    ###########################################################################################
    # Fig 3G : tuning curves during wake opto
    ###########################################################################################
    if f == 'OPTO_WAKE':
        st, gr, sd = 'lmn', 'opto', 'ipsi'
        si = data['allsi'][st][gr][sd]
        index = np.intersect1d(si['pre'][si['pre'] > 0.2].index.values, si['opto'][si['opto'] > 0.2].index.values)

        tcs = {}
        peaks = {}
        for k in ['opto', 'pre']:
            alltc = data['alltc'][st][gr][sd][k]
            peaks[k] = alltc.idxmax()
            tcs[k] = centerTuningCurves_with_peak(alltc[index])

        tcs2 = {k: tcs[k] / tcs['pre'].max() for k in ['opto', 'pre']}
        sheets["Fig3G_tuning_mean"] = pd.DataFrame({"HD (deg)": np.rad2deg(tcs2['pre'].index.values),
                                                    "mean norm. rate, stimulation": tcs2['opto'].mean(1).values,
                                                    "mean norm. rate, no stimulation": tcs2['pre'].mean(1).values})
        readme.append(("Fig3G_tuning_mean",
                       f"Wakefulness: centered tuning curves of LMN HD neurons (spatial information > 0.2 in both conditions) "
                       f"with (red) and without (dashed) PSB stimulation, normalized to the peak without stimulation; mean across n = {len(index)} neurons."))

        widths = {}
        for k in ['opto', 'pre']:
            tmp = tcs[k] / tcs[k].loc[0]
            tmp = tmp - 0.5
            widths[k] = np.rad2deg(np.abs(np.abs(tmp.loc[:0]).idxmin()) + np.abs(np.abs(tmp.loc[0:]).idxmin()))
        widths = pd.DataFrame(widths)
        maxpeaks = pd.DataFrame({"opto": tcs['opto'].max(), "pre": tcs['pre'].max()})
        # As in main_pyna_fig3.py, the peak angle difference uses all neurons (not only `index`)
        peaks = pd.DataFrame(peaks)

        d = peaks['opto'] - peaks['pre']
        d[d > np.pi] = 2 * np.pi - d[d > np.pi]
        d[d < -np.pi] = d[d < -np.pi] + 2 * np.pi
        d = d / np.pi

        df = pd.DataFrame({"neuron": index,
                           "width no stim. (deg)": widths['pre'].loc[index].values,
                           "width stim. (deg)": widths['opto'].loc[index].values,
                           "max FR no stim. (Hz)": maxpeaks['pre'].loc[index].values,
                           "max FR stim. (Hz)": maxpeaks['opto'].loc[index].values})
        df["Width norm. diff."] = (df["width stim. (deg)"] - df["width no stim. (deg)"]) / (df["width stim. (deg)"] + df["width no stim. (deg)"])
        df["Max FR norm. diff."] = (df["max FR stim. (Hz)"] - df["max FR no stim. (Hz)"]) / (df["max FR stim. (Hz)"] + df["max FR no stim. (Hz)"])
        sheets["Fig3G_tuning_params"] = df
        sheets["Fig3G_peak_angle"] = pd.DataFrame({"neuron": peaks.index.values,
                                                   "peak no stim. (deg)": np.rad2deg(peaks['pre'].values),
                                                   "peak stim. (deg)": np.rad2deg(peaks['opto'].values),
                                                   "Delta ang. (/pi)": d.values})

        stats = []
        for label, x in zip(["Width", "Max FR"], [widths.loc[index], maxpeaks.loc[index]]):
            zw, p = scipy.stats.wilcoxon(x['pre'], x['opto'])
            stats.append(f"{label}: W = {zw:.1f}, {fmt_p(p)}")
        readme.append(("Fig3G_tuning_params",
                       f"Wakefulness: width and max firing rate of each LMN HD neuron without and with PSB stimulation (n = {len(index)} neurons); "
                       f"bottom plot shows (stim - no stim) / (stim + no stim). "
                       f"Wilcoxon signed-rank tests, stim vs no stim (two-sided): " + "; ".join(stats) + "."))
        zw, p = scipy.stats.wilcoxon(peaks['pre'], peaks['opto'])
        readme.append(("Fig3G_peak_angle",
                       f"Wakefulness: preferred direction of each LMN neuron without and with PSB stimulation, and their difference (/pi) "
                       f"shown in the bottom plot (all n = {len(peaks)} neurons). "
                       f"Wilcoxon signed-rank test, stim vs no stim (two-sided): W = {zw:.1f}, {fmt_p(p)}."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("Fig3", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

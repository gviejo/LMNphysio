# -*- coding: utf-8 -*-
# Export the Source Data of Figure 5 (population panels only, no example neuron) to Source_Data.xlsx.
# Computations mirror main_pyna_fig5.py.
import numpy as np
import pandas as pd
import scipy.stats
from scipy.ndimage import gaussian_filter
import _pickle as cPickle
import os

from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []

names = {'adn': "ADN", 'lmn': "LMN"}
epochs = {'wak': 'Wake', 'sws': 'Sleep'}


def fmt_p(p):
    return f"p = {p:.3g}" if p > 0 else "p < 1e-300"


pisi = {'adn': cPickle.load(open(os.path.join(dropbox_path, 'PISI_ADN.pickle'), 'rb')),
        'lmn': cPickle.load(open(os.path.join(dropbox_path, 'PISI_LMN.pickle'), 'rb'))}

###############################################################################################
# Fig 5B (top) : normalized tuning curves
###############################################################################################
for e in ['wak', 'sws']:
    df = {}
    n = {}
    for st in ['adn', 'lmn']:
        tc = pisi[st]['tc_' + e]
        tc = tc / tc.max(0)
        df["HD (deg)"] = np.rad2deg(tc.index.values)
        df[f"{names[st]} mean rate (%)"] = 100 * tc.mean(1).values
        df[f"{names[st]} SD (%)"] = 100 * tc.std(1).values
        n[st] = tc.shape[1]
    sheets[f"Fig5B_tuning_{epochs[e]}"] = pd.DataFrame(df)
    readme.append((f"Fig5B_tuning_{epochs[e]}",
                   f"{epochs[e]}: centered tuning curves normalized to their maximum, mean +/- SD across "
                   f"n = {n['adn']} ADN and n = {n['lmn']} LMN neurons."))

###############################################################################################
# Fig 5B (middle) : ISI distribution as a function of centered HD
###############################################################################################
cut = -40
for st in ['adn', 'lmn']:
    for e in ['wak', 'sws']:
        bins = pisi[st]['bins'][0:cut]
        tmp = pisi[st][e].mean(0)[0:cut]
        tmp2 = np.hstack((tmp, tmp, tmp))
        tmp2 = gaussian_filter(tmp2, sigma=(1, 1))
        tmp3 = tmp2[:, tmp.shape[1]:tmp.shape[1] * 2]
        tmp3 = tmp3 / tmp3.sum(0)
        tmp3 = tmp3 * 100.0
        hd = np.linspace(-180, 180, tmp3.shape[1])
        df = pd.DataFrame(tmp3, columns=[f"{h:.0f} deg" for h in hd])
        df.insert(0, "ISI bin start (s)", bins[0:-1])
        sheets[f"Fig5B_ISI_HD_{names[st]}_{epochs[e]}"] = df
        readme.append((f"Fig5B_ISI_HD_{names[st]}_{epochs[e]}",
                       f"{names[st]}, {epochs[e]}: distribution of ISI (rows, log-spaced bins) as a function of centered HD (columns), "
                       f"averaged over n = {pisi[st][e].shape[0]} neurons, Gaussian smoothing sigma = 1 bin, "
                       f"normalized to 100% per HD bin (color scale, %)."))

###############################################################################################
# Fig 5B (right) : ISI distribution averaged over HD
###############################################################################################
for st in ['adn', 'lmn']:
    bins = pisi[st]['bins'][0:cut]
    df = {"ISI bin start (s)": bins[0:-1]}
    for e in ['wak', 'sws']:
        tmp = pisi[st][e].mean(0)[0:cut]
        y = tmp.mean(1)
        df[f"{epochs[e]} (%)"] = (y / y.sum()) * 100
    sheets[f"Fig5B_ISI_{names[st]}"] = pd.DataFrame(df)
    readme.append((f"Fig5B_ISI_{names[st]}",
                   f"{names[st]}: ISI distribution averaged over HD bins and neurons (n = {pisi[st]['wak'].shape[0]} neurons), in % per bin, wake and sleep."))

###############################################################################################
# Fig 5D : bimodality of ISI distributions
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'All_ISI.pickle'), 'rb'))
pr2 = data['pr2']
rows = []
for e in ['wak', 'sws']:
    for st in ['adn', 'lmn']:
        x = pr2[st][e].dropna()
        rows.append(pd.DataFrame({"epoch": epochs[e], "region": names[st], "neuron": x.index.values, "bimodality (pR2)": x.values}))
sheets["Fig5D"] = pd.concat(rows, ignore_index=True)

stats = []
for e in ['wak', 'sws']:
    tmp = [pr2[st][e] for st in ['adn', 'lmn']]
    zw, p = scipy.stats.mannwhitneyu(tmp[0].dropna(), tmp[1].dropna())
    stats.append(f"{epochs[e]}: n = {len(tmp[0].dropna())} ADN, n = {len(tmp[1].dropna())} LMN neurons, "
                 f"mean ADN {tmp[0].mean():.4f}, mean LMN {tmp[1].mean():.4f}, U = {zw:.1f}, {fmt_p(p)}")
readme.append(("Fig5D",
               "Bimodality of the log-ISI distribution of each neuron (pseudo-R2 of a 2- vs 1-component Gaussian mixture), "
               "ADN vs LMN; dot = mean. Mann-Whitney U test (two-sided). " + "; ".join(stats) + "."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("Fig5", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

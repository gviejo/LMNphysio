# -*- coding: utf-8 -*-
# Export the Source Data of Supplementary Figure 4 (LMN-ADN connected pairs) to Source_Data.xlsx.
# Computations mirror main_pyna_supp4.py. Panel A (example pairs) is not exported.
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


data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN-ADN.pickle'), 'rb'))
index = data['zorder'].index.values
order = data['order']

###############################################################################################
# Supp Fig 4B : angular difference between tuning curves of connected pairs
###############################################################################################
tmp = data['angdiff'][index]
h, b = np.histogram(tmp, np.linspace(-np.pi, np.pi, 22), range=(-np.pi, np.pi))
sheets["SuppFig4B_hist"] = pd.DataFrame({"bin start (deg)": np.rad2deg(b[:-1]), "bin end (deg)": np.rad2deg(b[1:]),
                                         "count": h, "proportion (%)": h / h.sum() * 100})
readme.append(("SuppFig4B_hist",
               f"Histogram of the angular difference between the preferred directions of LMN and ADN neurons "
               f"in LMN->ADN pairs with Z > 3 (n = {len(tmp)} pairs), in % of pairs per bin."))

###############################################################################################
# Supp Fig 4C : |angular difference| vs Z
###############################################################################################
angdiff = np.abs(data['angdiff'][order])
z = data['zorder'][order]
r, p = scipy.stats.pearsonr(angdiff.values, z.values)
m, b = np.polyfit(angdiff.values, z.values, 1)
sheets["SuppFig4BC_pairs"] = pd.DataFrame({
    "LMN neuron": [k[0] for k in order],
    "ADN neuron": [k[1] for k in order],
    "ang. diff. (rad)": data['angdiff'][order].values,
    "|ang. diff.| (rad)": angdiff.values,
    "Z": z.values,
})
readme.append(("SuppFig4BC_pairs",
               f"Each LMN->ADN pair with Z > 3 (n = {len(order)} pairs): angular difference between preferred directions "
               f"(radians; used in B and C) and Z of the cross-correlogram peak (C)."))
sheets["SuppFig4C"] = pd.DataFrame({"n pairs": [len(order)], "Pearson r": [r], "p (two-sided)": [p],
                                    "fit slope": [m], "fit intercept": [b]})
readme.append(("SuppFig4C",
               f"Correlation between |ang. diff.| and Z (red line = linear fit). n = {len(order)} pairs, "
               f"Pearson r = {r:.4f}, {fmt_p(p)} (two-sided)."))

###############################################################################################
# Supp Fig 4D : lag of the cross-correlogram peak
###############################################################################################
peak = data['zcc']['sws'][order].loc[0.001:0.008].idxmax()
sheets["SuppFig4D_lags"] = pd.DataFrame({"LMN neuron": [k[0] for k in order],
                                         "ADN neuron": [k[1] for k in order],
                                         "peak lag (ms)": peak.values * 1e3})
hist_, bin_edges = np.histogram(peak.values, bins=np.linspace(0, 0.01, 25), range=(0, 0.01))
sheets["SuppFig4D_hist"] = pd.DataFrame({"bin start (ms)": bin_edges[:-1] * 1e3, "bin end (ms)": bin_edges[1:] * 1e3,
                                         "count": hist_, "proportion (%)": hist_ / hist_.sum() * 100})
readme.append(("SuppFig4D_lags",
               f"Lag of the non-REM cross-correlogram peak (searched between 1 and 8 ms) of each LMN->ADN pair "
               f"(n = {len(order)} pairs). Mean lag = {peak.mean() * 1e3:.2f} ms."))
readme.append(("SuppFig4D_hist", "Histogram of peak lags, in % of pairs per bin."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("SuppFig4", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

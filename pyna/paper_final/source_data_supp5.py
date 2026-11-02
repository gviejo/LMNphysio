# -*- coding: utf-8 -*-
# Export the Source Data of Supplementary Figure 5 (PSB HD cells, PSB-LMN angular offset) to Source_Data.xlsx.
# Panel A mirrors main_pyna_supp5.py. Panel B is histology.
# Panel C is not produced by main_pyna_supp5.py: it mirrors the commented "Angular differences" block of main_pyna_fig2.py.
import sys
import numpy as np
import pandas as pd
import _pickle as cPickle
import os

sys.path.append("../")
from functions import centerTuningCurves_with_peak
from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []

###############################################################################################
# Supp Fig 5A : PSB HD cell selection and tuning curves
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'All_TC.pickle'), 'rb'))
alltc = data['alltc']
allinfo = data['allinfo']

idx = allinfo[allinfo['location'] == 'psb'].index.values
info = allinfo.loc[idx]
tokeep = info[(info['rate'] > 1.0) & (info['SI'] > 0.1) & (info['tokeep'] == 1)].index.values

sheets["SuppFig5A_selection"] = pd.DataFrame({
    "neuron": idx,
    "rate (Hz)": info['rate'].values,
    "HD info. (bits/spk)": info['SI'].values,
    "selected HD neuron": np.isin(idx, tokeep).astype(int),
})
readme.append(("SuppFig5A_selection",
               f"PSB: mean firing rate vs HD information of each neuron (n = {len(idx)} neurons). "
               f"Selected HD neurons (rate > 1 Hz, HD info. > 0.1 bits/spk): n = {len(tokeep)} (colored dots)."))

tc = centerTuningCurves_with_peak(alltc[tokeep])
tc = tc / tc.max()
sheets["SuppFig5A_tuning_mean"] = pd.DataFrame({
    "angle (deg)": np.rad2deg(tc.index.values),
    "mean norm. rate": tc.mean(1).values,
    "SD": tc.std(1).values,
})
readme.append(("SuppFig5A_tuning_mean",
               f"PSB: centered tuning curves normalized to their maximum, mean +/- SD across n = {tc.shape[1]} HD neurons."))

heat = tc.T.copy()
heat.columns = [f"{a:.1f} deg" for a in np.rad2deg(tc.index.values)]
heat.insert(0, "neuron", tc.columns.values)
sheets["SuppFig5A_tuning_heatmap"] = heat
readme.append(("SuppFig5A_tuning_heatmap",
               f"PSB: centered normalized tuning curve of each HD neuron (n = {tc.shape[1]}; one row per neuron, color scale = rate)."))

###############################################################################################
# Supp Fig 5C : angular offset of PSB->LMN pairs
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN-PSB.pickle'), 'rb'))
angdiff = data['angdiff']
order = data['order']

sheets["SuppFig5C_pairs"] = pd.DataFrame({"neuron 1": [k[0] for k in order],
                                          "neuron 2": [k[1] for k in order],
                                          "ang. offset (deg)": np.rad2deg(angdiff[order].values)})
readme.append(("SuppFig5C_pairs",
               f"Angular offset between the preferred directions of PSB and LMN neurons for the PSB->LMN pairs "
               f"with a positive-lag peak (n = {len(order)} pairs, same pairs as Fig. 2F)."))

bins = np.linspace(-np.pi, np.pi, 33)
counts, bin_edges = np.histogram(angdiff[order], bins=bins)
sheets["SuppFig5C_hist"] = pd.DataFrame({"bin start (deg)": np.rad2deg(bin_edges[:-1]),
                                         "bin end (deg)": np.rad2deg(bin_edges[1:]),
                                         "count": counts, "proportion (%)": counts / counts.sum() * 100})
readme.append(("SuppFig5C_hist", "Histogram of angular offsets (32 bins), in % of pairs per bin."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("SuppFig5", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

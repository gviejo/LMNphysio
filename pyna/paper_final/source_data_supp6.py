# -*- coding: utf-8 -*-
# Export the Source Data of Supplementary Figure 6 (LMN during UP/DOWN states) to Source_Data.xlsx.
# Computations mirror main_pyna_supp6.py.
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


def fmt_p(p):
    return f"p = {p:.3g}" if p > 0 else "p < 1e-300"


###############################################################################################
# Supp Fig 6A, B : LMN firing rate UP vs DOWN
###############################################################################################
frates = cPickle.load(open(os.path.join(dropbox_path, 'LMN_UP_DOWN.pickle'), 'rb'))['frates']
stat_ud, pval_ud = scipy.stats.wilcoxon(frates['up'], frates['down'])

sheets["SuppFig6AB"] = pd.DataFrame({"neuron": frates.index.values,
                                     "rate Up (Hz)": frates['up'].values,
                                     "rate Down (Hz)": frates['down'].values,
                                     "modulation (down - up) / up": frates['delta'].values})
readme.append(("SuppFig6AB",
               f"Firing rate of each LMN neuron during UP and DOWN states (A) and modulation index (B). n = {len(frates)} neurons. "
               f"A: mean +/- SEM, Up {frates['up'].mean():.3f} +/- {frates['up'].sem():.3f} Hz, "
               f"Down {frates['down'].mean():.3f} +/- {frates['down'].sem():.3f} Hz; "
               f"Wilcoxon signed-rank test (two-sided): W = {stat_ud:.1f}, {fmt_p(pval_ud)}. "
               f"B: median modulation index = {np.median(frates['delta']):.4f} (orange line)."))

bins = np.linspace(frates['delta'].min(), frates['delta'].max(), 25)
h, b = np.histogram(frates['delta'], bins=bins)
sheets["SuppFig6B_hist"] = pd.DataFrame({"bin start": b[:-1], "bin end": b[1:], "neuron count": h})
readme.append(("SuppFig6B_hist", "Histogram of the modulation index (neuron count per bin)."))

###############################################################################################
# Supp Fig 6C, D : PSB and LMN cross-correlograms around DOWN and UP onsets, and their derivative
###############################################################################################
cc_data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN_DOWN.pickle'), 'rb'))
lags = cc_data['lags']

for event in ['DOWN', 'UP']:
    df_cc = {"time (ms)": lags}
    df_d = {"time (ms)": lags}
    n = {}
    for region in ['PSB', 'LMN']:
        data = cc_data[f"{region.lower()}_cc_{event.lower()}"]
        n[region] = data.shape[1]
        df_cc[f"{region} mean norm. rate"] = data.mean(axis=1).values
        df_cc[f"{region} SEM"] = data.sem(axis=1).values

        m = gaussian_filter1d(data.mean(axis=1).values, sigma=5)
        dm = np.gradient(m, lags)
        dm = (dm - dm.mean()) / dm.std()
        s = np.abs(np.gradient(gaussian_filter1d(data.sem(axis=1).values, sigma=5), lags)) / dm.std()
        df_d[f"{region} rate of change (z)"] = dm
        df_d[f"{region} error"] = s

    sheets[f"SuppFig6C_{event}"] = pd.DataFrame(df_cc)
    readme.append((f"SuppFig6C_{event}",
                   f"Normalized cross-correlogram of PSB and LMN activity around {event} onset, mean +/- SEM across "
                   f"n = {n['PSB']} sessions (columns of the data)."))
    sheets[f"SuppFig6D_{event}"] = pd.DataFrame(df_d)
    readme.append((f"SuppFig6D_{event}",
                   f"Derivative of the mean cross-correlogram around {event} onset (mean smoothed with a Gaussian, sigma = 5 bins), "
                   f"z-scored. Error = derivative of the smoothed SEM divided by the SD of the derivative (shaded area)."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("SuppFig6", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

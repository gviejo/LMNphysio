# -*- coding: utf-8 -*-
# Export the Source Data of Supplementary Figure 10 (behavior during PSB stimulation) to Source_Data.xlsx.
# Computations mirror main_pyna_supp8.py.
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import _pickle as cPickle
import os

from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []


def fmt_p(p):
    return f"p = {p:.3g}" if p > 0 else "p < 1e-300"


data = cPickle.load(open(os.path.join(dropbox_path, 'OPTO_behavior_characterization.pickle'), 'rb'))
df = data['df']
cc_all = data['cc_all']

sides = {'ipsi': "Ipsilateral", 'bilateral': "Bilateral"}

for P, col_off, col_on, label, cc_key in [
        ("A", 'ang_off', 'ang_on', 'angular velocity (rad/s)', 'ang'),
        ("B", 'lin_off', 'lin_on', 'linear velocity (m/s)', 'lin')]:

    ###########################################################################################
    # Left : session velocity, stimulation OFF vs ON
    ###########################################################################################
    rows = []
    stats = []
    for sd in ['ipsi', 'bilateral']:
        sub = df[(df['side'] == sd) & (df['group'] == 'opto')].dropna(subset=[col_on, col_off])
        off_vals = sub[col_off].values
        on_vals = sub[col_on].values
        rows.append(pd.DataFrame({"stimulation": sides[sd], "session": sub['session'].values,
                                  "recorded area": sub['area'].str.upper().values,
                                  f"OFF {label}": off_vals, f"ON {label}": on_vals}))
        z, pval = wilcoxon(off_vals, on_vals)
        stats.append(f"{sides[sd]}: n = {len(sub)} sessions, OFF {off_vals.mean():.4f} +/- {off_vals.std(ddof=1) / np.sqrt(len(off_vals)):.4f}, "
                     f"ON {on_vals.mean():.4f} +/- {on_vals.std(ddof=1) / np.sqrt(len(on_vals)):.4f}, W = {z:.1f}, {fmt_p(pval)}")
    sheets[f"SuppFig10{P}_sessions"] = pd.concat(rows, ignore_index=True)
    readme.append((f"SuppFig10{P}_sessions",
                   f"Mean {label} of the animal outside (OFF) and during (ON) PSB stimulation, per Chrimson session. "
                   f"Mean +/- SEM across sessions. Wilcoxon signed-rank test OFF vs ON (two-sided): " + "; ".join(stats) + "."))

    ###########################################################################################
    # Right : velocity around stimulation onset
    ###########################################################################################
    out = {}
    n = {}
    for sd in ['ipsi', 'bilateral']:
        data_cc = cc_all['opto'][sd][cc_key]
        out["time from stimulation (s)"] = data_cc.index.values
        out[f"{sides[sd]} mean"] = data_cc.mean(axis=1).values
        out[f"{sides[sd]} SEM"] = data_cc.std(axis=1).values / np.sqrt(data_cc.shape[1])
        n[sd] = data_cc.shape[1]
    sheets[f"SuppFig10{P}_time"] = pd.DataFrame(out)
    readme.append((f"SuppFig10{P}_time",
                   f"Mean {label} around PSB stimulation (stimulation from 0 to 10 s), mean +/- SEM across "
                   f"n = {n['ipsi']} ipsilateral and n = {n['bilateral']} bilateral sessions."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("SuppFig10", sheets, dict(readme))
for name, df_ in sheets.items():
    print(name, df_.shape)
for name, desc in sorted(readme):
    print(name, ":", desc)

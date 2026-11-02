# -*- coding: utf-8 -*-
# Export the Source Data of Figure 6 (panel D only; B and C show a single simulation) to Source_Data.xlsx.
# Computations mirror main_pyna_fig6.py.
import numpy as np
import pandas as pd
import _pickle as cPickle
import os

from source_data_utils import write_figure_sheet

dropbox_path = "/home/gviejo/Dropbox/LMNphysio/data"

sheets = {}
readme = []

names = {'adn': "ADN", 'lmn': "LMN"}

###############################################################################################
# Fig 6D (Observed) : population coherence per session, sleep vs opto
###############################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
r_lmn = data['pearsonr']
data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'rb'))
r_adn = data['pearsonr']
allr_sess = {'adn': r_adn, 'lmn': r_lmn}

data = cPickle.load(open(os.path.join(dropbox_path, 'OPTO_SLEEP.pickle'), 'rb'))
opto = {'lmn': data['corr']['lmn']['opto']['ipsi']['opto'].astype(float),
        'adn': data['corr']['adn']['opto']['bilateral']['opto'].astype(float)}
opto_side = {'lmn': "ipsilateral", 'adn': "bilateral"}

summary = []
for st in ['adn', 'lmn']:
    rows = [pd.DataFrame({"condition": "Sleep", "session": allr_sess[st]['sws'].index.values,
                          "pop. coherence (r)": allr_sess[st]['sws'].values.astype(float)}),
            pd.DataFrame({"condition": "Opto.", "session": opto[st].index.values,
                          "pop. coherence (r)": opto[st].values})]
    sheets[f"Fig6D_observed_{names[st]}"] = pd.concat(rows, ignore_index=True)
    for cond, x in zip(["Sleep", "Opto."], [allr_sess[st]['sws'].astype(float), opto[st]]):
        summary.append({"region": names[st], "condition": cond, "n sessions": len(x), "mean (r)": x.mean(), "SD (r)": x.std()})
    readme.append((f"Fig6D_observed_{names[st]}",
                   f"Observed {names[st]} population coherence per session: non-REM sleep (n = {len(allr_sess[st])} sessions, as in Fig. 1J) "
                   f"and non-REM sleep with {opto_side[st]} PSB stimulation (n = {len(opto[st])} sessions)."))

sheets["Fig6D_observed_mean"] = pd.DataFrame(summary)
readme.append(("Fig6D_observed_mean", "Observed population coherence: mean +/- SD across sessions (dots and error bars)."))

###############################################################################################
# Fig 6D (Model) : population coherence in the model
###############################################################################################
popcoh_seeds = cPickle.load(open("/home/gviejo/Dropbox/LMNphysio/model/model_popcoh_seeds.pickle", 'rb'))['popcoh']

df = popcoh_seeds.reset_index()
df.columns = ["seed", "ADN 'Sleep' (r)", "ADN 'Opto.' (r)", "LMN 'Sleep' (r)", "LMN 'Opto.' (r)"]
sheets["Fig6D_model_seeds"] = df
readme.append(("Fig6D_model_seeds",
               f"Model population coherence for each simulation (n = {len(df)} simulations with different random seeds): "
               f"Pearson r between the pairwise correlations of all simulated neuron pairs (630 LMN pairs, 64620 ADN pairs) "
               f"in 'Wake' and in 'Sleep' or 'Opto.'."))

summary = []
for st in ['adn', 'lmn']:
    for k, cond in zip(['sws', 'opto'], ["'Sleep'", "'Opto.'"]):
        x = popcoh_seeds[f"{st}_{k}"]
        summary.append({"region": names[st], "condition": cond, "n simulations": len(x), "mean (r)": x.mean(), "SD (r)": x.std()})

sheets["Fig6D_model_mean"] = pd.DataFrame(summary)
readme.append(("Fig6D_model_mean", "Model population coherence: mean +/- SD across simulations (dots and error bars)."))

###############################################################################################
# WRITE
###############################################################################################
write_figure_sheet("Fig6", sheets, dict(readme))
for name, df in sheets.items():
    print(name, df.shape)
print(sheets["Fig6D_observed_mean"])
print(sheets["Fig6D_model_mean"])

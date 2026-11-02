# LMNphysio

Analysis code for:

> Viejo G., Skromne Carrasco S., Peyrache A. Manuscript submitted to *Nature Communications* (NCOMMS-26-016525).

The code analyses extracellular recordings of head-direction (HD) cells in the lateral mammillary nucleus (LMN), the anterodorsal thalamic nucleus (ADN) and the postsubiculum (PSB), during wakefulness and sleep, with and without optogenetic silencing of PSB. It also contains the network model of Fig. 6.

## Repository layout

| Folder | Content |
|---|---|
| `pyna/paper_final/` | Scripts that draw every figure of the paper and write the Source Data file |
| `pyna/` | Analysis scripts (`main_pyna_*.py`) that compute the intermediate results from the NWB files |
| `pyna/OPTO/` | Analysis of the optogenetic experiments |
| `pyna/ISI/` | Inter-spike-interval analysis and decoding (Fig. 5) |
| `pyna/review/` | Analyses added during peer review |
| `pyna/examples/` | Scripts that select and save the example sessions shown in the figures |
| `pyna/functions.py` | Shared helper functions |
| `model/` | Network model (Fig. 6, Supp. Fig. 11) |
| `pyna/paper20xx/`, `pyna/cosyne20xx/`, `python/`, ... | Earlier versions, posters and abstracts. Not needed to reproduce the paper |

## Installation

The figures were made with Python 3.11 and:

```
pynapple 0.11.3    numpy 2.3.5          pandas 2.3.3     scipy 1.16.3
matplotlib 3.10.7  scikit-learn 1.7.2   seaborn 0.13.2   numba
pynwb 3.1.2        pycircstat 0.0.2     colorcet 3.1.0   hsluv 5.0.4
openpyxl 3.1.5     pyyaml 6.0.3
```

```bash
conda create -n lmnphysio python=3.11
conda activate lmnphysio
pip install pynapple==0.11.3 numpy==2.3.5 pandas==2.3.3 scipy==1.16.3 \
    matplotlib==3.10.7 scikit-learn==1.7.2 seaborn==0.13.2 numba pynwb==3.1.2 \
    pycircstat==0.0.2 colorcet==3.1.0 hsluv==5.0.4 openpyxl==3.1.5 pyyaml==6.0.3
```

`req.txt` at the root is an old (Python 3.6) environment and is not used by the current code.

Spike sorting was done with Kilosort 4 and manually curated with Klusters. 


## Reproducing the figures

Reproduction has two steps:

1. Run the analysis scripts. They read the NWB files and write the pickle files.
2. Run the figure scripts in `pyna/paper_final/`. They read only the pickle files, so step 1 is not needed if you have the pickle files.

```bash
cd pyna/paper_final
python main_pyna_fig1.py       # writes paperfinal/fig1.pdf
python source_data_fig1.py     # writes the Fig1 sheet of paperfinal/Source_Data.xlsx
```

Each `main_pyna_<fig>.py` draws one figure. Each `source_data_<fig>.py` writes the matching sheet of `Source_Data.xlsx` and prints the statistics. The figure scripts load the Arial `.ttf` files from the `paperfinal` folder; without them, the text falls back to Liberation Sans or DejaVu Sans.

### Which script makes which file

| Figure | Figure script (`pyna/paper_final/`) | Intermediate files | Analysis script that writes them |
|---|---|---|---|
| Fig. 1 | `main_pyna_fig1.py` | `All_correlation_ADN.pickle` | `pyna/main_pyna_ADN_correlation.py` |
| | | `All_correlation_LMN.pickle` | `pyna/main_pyna_LMN_correlation.py` |
| | | `All_CC_ADN.pickle`, `All_CC_LMN.pickle` | `pyna/main_pyna_ADN_CC.py`, `pyna/main_pyna_LMN_CC.py` |
| | | `CC_LMN-ADN.pickle` | `pyna/main_pyna_LMN_ADN_CC.py` |
| | | `DATA_FIG_LMN_ADN_A5011-201014A.pickle`, `projections_KPCA_LMN_ADN_v2.pickle` | `pyna/main_pyna_LMN_ADN_rings2.py` |
| Fig. 2 | `main_pyna_fig2.py` | `CC_LMN-PSB.pickle` | `pyna/main_pyna_LMN_PSB_CC.py` |
| | | `CORR_LMN-PSB_UP_DOWN.pickle` | `pyna/main_pyna_LMN_UP_DOWN_modulation.py` |
| | | `DATA_FIG_LMN_PSB_A3019-220701A.pickle` | `pyna/examples/main_pyna_search_exemple_LMN_PSB.py` |
| Fig. 3 | `main_pyna_fig3.py` | `OPTO_WAKE.pickle` | `pyna/OPTO/main_pyna_OPTO_WAKE.py` |
| | | `OPTO_SLEEP.pickle` | `pyna/OPTO/main_pyna_OPTO_SLEEP.py` |
| Fig. 4 | `main_pyna_fig4.py` | `OPTO_WAKE.pickle`, `OPTO_SLEEP.pickle` | as Fig. 3 |
| Fig. 5 | `main_pyna_fig5.py` | `All_ISI.pickle` | `pyna/ISI/main_pyna_ISI.py` |
| | | `ALL_LOG_ISI.pickle` | `pyna/ISI/main_pyna_ISI_analysis.py` |
| | | `PISI_ADN.pickle`, `PISI_LMN.pickle` | `pyna/ISI/main_pyna_ISI_ADN_decoding.py`, `pyna/ISI/main_pyna_ISI_LMN_decoding.py` |
| Fig. 6 | `main_pyna_fig6.py` | `All_correlation_*.pickle`, `OPTO_SLEEP.pickle` | as Figs. 1 and 3 |
| | | `model/model_rings.pickle` | `model/main_model4_rings.py` (commit `217321e`) |
| | | `model/model_popcoh_seeds.pickle` | `model/main_model4_rings_seeds.py` (1000 seeds) |
| Supp. Fig. 2 | `main_pyna_supp2.py` | `All_TC.pickle` | `pyna/main_pyna_ALL_TC.py` |
| | | `random_LMN_comparison.pickle` | `pyna/review/main_generate_random_LMN.py` |
| Supp. Fig. 3 | `main_pyna_supp3.py` | `All_TC.pickle`, `random_LMN_comparison.pickle` | as Supp. Fig. 2 |
| Supp. Fig. 4 | `main_pyna_supp4.py` | `All_TC.pickle`, `CC_LMN-ADN.pickle` | as Fig. 1 and Supp. Fig. 2 |
| Supp. Fig. 5 | `main_pyna_supp5.py` | `OPTO_PSB.pickle` | `pyna/OPTO/main_pyna_OPTO_PSB.py` |
| | | `DATA_FIG_PSB_SLEEP_A8054-230718A.pickle` | `pyna/examples/main_pyna_search_example_PSB_OPTO.py` |
| Supp. Fig. 6 | `main_pyna_supp6.py` | `LMN_UP_DOWN.pickle` | `pyna/review/main_pyna_LMN_UP_DOWN_review.py` |
| | | `CC_LMN_DOWN.pickle` | `pyna/review/main_cc_LMN_DOWN.py` |
| | | `CORR_LMN-PSB_UP_DOWN.pickle` | as Fig. 2 |
| Supp. Fig. 7 | `main_pyna_supp7.py` | `OPTO_PSB.pickle`, `All_TC.pickle` | as Supp. Figs. 5 and 2 |
| Supp. Fig. 10 | `main_pyna_supp8.py` | `OPTO_behavior_characterization.pickle` | `pyna/review/main_pyna_OPTO_behavior_characterization.py` |
| Supp. Fig. 11 | `main_pyna_supp11.py` | `model/model_rings.pickle`, `model/model_popcoh_seeds.pickle` | as Fig. 6 |



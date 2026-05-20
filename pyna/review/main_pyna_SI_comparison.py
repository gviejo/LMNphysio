"""
Compare Spatial Information, directional tuning width, and mean vector length
between ADN and LMN HD cells across all LMN-ADN datasets.
"""
# %%
import numpy as np
import pandas as pd
import pynapple as nap
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu
from pycircstat.descriptive import mean as circmean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from functions import *

# %%
###############################################################################################
# GENERAL infos
###############################################################################################
if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'

datasets = np.genfromtxt(
    os.path.join(data_directory, 'datasets_LMN_ADN.list'),
    delimiter='\n', dtype=str, comments='#'
)
datasets = np.unique(datasets)


# %%
def compute_mean_vector_length(tuning_curve):
    """Resultant vector length of a circular tuning curve (0–1)."""
    angles = tuning_curve.index.values
    rates  = tuning_curve.values
    total  = rates.sum()
    if total == 0:
        return np.nan
    return np.abs(np.sum(rates * np.exp(1j * angles))) / total


def compute_tuning_width(tuning_curve):
    """
    Half-width at half-maximum of the tuning curve, in degrees.
    Counts the arc of angle bins above half the peak firing rate.
    """
    rates    = tuning_curve.values
    peak     = rates.max()
    if peak == 0:
        return np.nan
    bin_width_deg = 360.0 / len(rates)
    return np.sum(rates >= peak / 2) * bin_width_deg


# %%
###############################################################################################
# LOOP OVER SESSIONS
###############################################################################################
records        = []
decode_records = []

for s in datasets:
    print(s)

    path     = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

    if not os.path.exists(filepath):
        continue

    nwb = nap.load_file(filepath)

    spikes = nwb['units']
    spikes = spikes.getby_threshold("rate", 1)

    # keep only ADN and LMN units
    if not hasattr(spikes, 'location'):
        nwb.close()
        continue
    spikes = spikes[spikes.location.isin(['adn', 'lmn'])]
    if len(spikes) == 0:
        nwb.close()
        continue

    # position
    columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    position = nap.TsdFrame(
        t=nwb['x'].t,
        d=np.column_stack([nwb[k].values for k in columns]),
        columns=columns,
        time_support=nwb['position_time_support']
    )

    nwb.close()

    ###############################################################################################
    # TUNING CURVES & CELL SELECTION
    ###############################################################################################
    tuning_curves = nap.compute_1d_tuning_curves(
        spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
        ep=position.time_support.loc[[0]]
    )
    tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

    SI = nap.compute_1d_mutual_info(
        tuning_curves, position['ry'],
        position.time_support.loc[[0]], minmax=(0, 2 * np.pi)
    )
    spikes.set_info(SI)
    spikes = spikes[spikes.SI > 0.1]

    if len(spikes) == 0:
        continue

    # half-epoch stability check
    wake2_ep = splitWake(position.time_support.loc[[0]])
    tokeep2  = []
    for i in range(2):
        tc_half = nap.compute_1d_tuning_curves(
            spikes, position['ry'], 120, minmax=(0, 2 * np.pi),
            ep=wake2_ep[i]
        )
        tc_half = smoothAngularTuningCurves(tc_half, 20, 4)
        tokeep, _ = findHDCells(tc_half)
        tokeep2.append(tokeep)
    tokeep = np.intersect1d(tokeep2[0], tokeep2[1])

    if len(tokeep) < 2:
        continue

    spikes      = spikes[tokeep]
    tuning_curves = tuning_curves[tokeep]

    ####
    # Decoding comparison — ADN only vs LMN only
    ####
    wake_ep_dec = position.time_support.loc[[0]]
    bin_size_dec = 0.1  # seconds

    for loc in ['adn', 'lmn']:
        group_idx = spikes.index[spikes.location == loc]
        if len(group_idx) < 5:
            continue
        group_sp = spikes[group_idx]

        # New-API tuning curves (xarray) required by decode_bayes
        tc_xr = nap.compute_tuning_curves(
            group_sp,
            position['ry'].restrict(wake_ep_dec),
            bins=120,
            epochs=wake_ep_dec,
        )

        # Bayesian decoding
        decoded, _ = nap.decode_bayes(
            tc_xr, group_sp, wake_ep_dec,
            bin_size=bin_size_dec,
            sliding_window_size=3,
        )

        # Circular decoding error (degrees)
        actual = decoded.value_from(position['ry'])
        err_rad = np.abs(np.angle(np.exp(1j * (decoded.values - actual.values))))
        mean_err_deg = float(np.degrees(np.nanmean(err_rad)))

        decode_records.append({
            'session':  s,
            'location': loc,
            'dec_error': mean_err_deg,
        })

    ###############################################################################################
    # METRICS PER NEURON
    ###############################################################################################
    for nid in spikes.index:
        loc = spikes.location[nid]
        if loc not in ('adn', 'lmn'):
            continue
        tc = tuning_curves[nid]
        records.append({
            'session':      s,
            'neuron':       nid,
            'location':     loc,
            'SI':           spikes.SI[nid],
            'MVL':          compute_mean_vector_length(tc),
            'tuning_width': compute_tuning_width(tc),
        })

df = pd.DataFrame(records)
df_decode = pd.DataFrame(decode_records)
print(f"\nTotal HD cells — ADN: {(df.location=='adn').sum()}  LMN: {(df.location=='lmn').sum()}")


# %%
###############################################################################################
# FIGURE
###############################################################################################
col_adn = '#5B9BD5'
col_lmn = '#ED7D31'

metrics = [
    ('SI',           'Spatial information (bits/spike)'),
    ('MVL',          'Mean vector length'),
    ('tuning_width', 'Tuning width (°)'),
]

adn = df[df.location == 'adn']
lmn = df[df.location == 'lmn']

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.subplots_adjust(wspace=0.45, hspace=0.45)
axes = axes.flatten()


def plot_adn_lmn(ax, adn_vals, lmn_vals, ylabel, n_adn=None, n_lmn=None):
    """Scatter + mean±SEM + connecting line + Mann-Whitney bracket."""
    jitter = 0.08
    np.random.seed(42)
    ax.scatter(np.random.normal(0, jitter, len(adn_vals)), adn_vals,
               color=col_adn, alpha=0.4, s=15, zorder=2)
    ax.scatter(np.random.normal(1, jitter, len(lmn_vals)), lmn_vals,
               color=col_lmn, alpha=0.4, s=15, zorder=2)

    means = []
    for x, vals, c in [(0, adn_vals, col_adn), (1, lmn_vals, col_lmn)]:
        m  = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
        means.append(m)
        ax.errorbar(x, m, yerr=se, fmt='o', color=c,
                    markersize=9, capsize=5, lw=2.5, zorder=5)
    ax.plot([0, 1], means, '-', color='k', lw=1.5, zorder=4)

    _, pval = mannwhitneyu(adn_vals, lmn_vals, alternative='two-sided')
    ymax = max(adn_vals.max(), lmn_vals.max())
    ypad = (ymax - min(adn_vals.min(), lmn_vals.min())) * 0.08
    ytop = ymax + ypad
    ax.plot([0, 0, 1, 1], [ytop * 0.97, ytop, ytop, ytop * 0.97], lw=1.2, color='k')
    pstr = f'p = {pval:.3g}' if pval >= 0.001 else 'p < 0.001'
    ax.text(0.5, ytop * 1.01, pstr, ha='center', va='bottom', fontsize=9)

    n_a = n_adn if n_adn is not None else len(adn_vals)
    n_l = n_lmn if n_lmn is not None else len(lmn_vals)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'ADN\n(n={n_a})', f'LMN\n(n={n_l})'], fontsize=10)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)


# Panels 1–3: per-neuron tuning metrics
for ax, (metric, ylabel) in zip(axes[:3], metrics):
    plot_adn_lmn(ax,
                 adn[metric].dropna().values,
                 lmn[metric].dropna().values,
                 ylabel)

# Panel 4: decoding error (per session)
ax = axes[3]
adn_dec = df_decode[df_decode.location == 'adn']['dec_error'].dropna().values
lmn_dec = df_decode[df_decode.location == 'lmn']['dec_error'].dropna().values
plot_adn_lmn(ax, adn_dec, lmn_dec, 'Decoding error (°)')

plt.suptitle('ADN vs LMN HD cells — tuning properties & decoding', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/SI_comparison_ADN_LMN.png"),
    dpi=300, bbox_inches='tight'
)
plt.show()
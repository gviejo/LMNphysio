import _pickle as cPickle
import os
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.pyplot import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from scipy.signal import find_peaks

dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")

psbdata = cPickle.load(open(dropbox_path + "/OPTO_PSB.pickle", "rb"))

# Opto manipulation
opto_mod = []
waveforms = []
for i, ep, sl, msl in zip(range(2), ['wake', 'sleep'], [slice(1, 10), slice(0.1, 1)], [slice(-5, -1), slice(-1, -0.1)]):
    for j, gr in enumerate(['opto', 'ctrl']):
        tmp = psbdata['allfr'][gr][ep]
        y = (tmp.loc[sl].mean().values - tmp.loc[msl].mean().values) / (
                    tmp.loc[sl].mean().values + tmp.loc[msl].mean().values)

        opto_mod.append(pd.Series(y, index=tmp.columns))

        waveforms.append(psbdata['allmwf'][gr][ep])

opto_mod = pd.concat(opto_mod, axis=0)
waveforms = pd.concat(waveforms, axis=1)

# waveforms = cPickle.load(open(os.path.join(dropbox_path, "meanwaveforms_psb_opto.pickle"), "rb"))
tmp = waveforms.values.reshape(32, 64, -1)
idx = np.argmax(np.sum(np.abs(tmp), 0), 0)
tmp = tmp[:, idx, np.arange(len(idx))]
tmp = tmp - np.nanmean(tmp, 0)
tmp = tmp / np.nanstd(tmp, 0)

mwf = pd.DataFrame(tmp, index=np.arange(32), columns=waveforms.columns)
mwf = mwf.dropna(axis=1)



info = []
for gr in ['opto']:
    for e in ['wake', 'sleep']:
        info.append(psbdata['allmeta'][gr][e])
info = pd.concat(info, axis=0)

tokeep = np.intersect1d(mwf.columns, info.index)
info = info.loc[tokeep]
mwf = mwf[tokeep]

through_to_peak = (mwf.idxmax() - mwf.idxmin())/30000
through_to_peak = through_to_peak[through_to_peak > 0.0]

tokeep = np.intersect1d(through_to_peak.index, info.index)

through_to_peak = through_to_peak.loc[tokeep]
info = info.loc[tokeep]

# exc_cells = through_to_peak[through_to_peak > 0.0003].index
# inh_cells = through_to_peak[through_to_peak <= 0.0003].index

exc_cells = tokeep[(through_to_peak > 0.0003) & (info['SI'] > 0.1)]
inh_cells = tokeep[(through_to_peak < 0.0003) & (info['SI'] < 0.1)]
# exc_cells = tokeep[info['SI'] > 0.5]
# inh_cells = tokeep[info['SI'] < 0.5]

unk_cells = np.setdiff1d(tokeep, np.union1d(exc_cells, inh_cells))


shared_idx = np.intersect1d(tokeep, opto_mod.index)
opto_mod = opto_mod.loc[shared_idx]

fig = figure(figsize=(10, 4), layout='constrained')
gs = GridSpec(1, 3, figure=fig, wspace=0.35)

ax_waveform = fig.add_subplot(gs[0, 0])
gs_merged = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1],
                                    hspace=0.05, height_ratios=[3, 1])
ax_merged  = fig.add_subplot(gs_merged[0])
ax_hist    = fig.add_subplot(gs_merged[1], sharex=ax_merged)
gs_opto = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.1)
ax_opto_exc = fig.add_subplot(gs_opto[0])
ax_opto_inh = fig.add_subplot(gs_opto[1], sharex=ax_opto_exc)

ax_merged.scatter(through_to_peak.loc[unk_cells], info.loc[unk_cells, 'SI'], color='grey', alpha=0.4, label='unk', zorder=1)
ax_merged.scatter(through_to_peak.loc[exc_cells], info.loc[exc_cells, 'SI'], color='royalblue', alpha=0.5, label='exc', zorder=2)
ax_merged.scatter(through_to_peak.loc[inh_cells], info.loc[inh_cells, 'SI'], color='tomato', alpha=0.5, label='inh', zorder=2)
ax_merged.axvline(0.0003, color='red', lw=1, ls='--')
ax_merged.axhline(0.5, color='red', lw=1, ls='--')
ax_merged.set_ylabel("SI")
ax_merged.tick_params(labelbottom=False)
ax_merged.legend()

ax_hist.hist(through_to_peak.values, bins=30, color='grey', alpha=0.7)
ax_hist.axvline(0.0003, color='red', lw=1, ls='--')
ax_hist.set_xlabel("Trough to peak (s)")
ax_hist.set_ylabel("Count")

exc_opto = np.intersect1d(exc_cells, shared_idx)
inh_opto = np.intersect1d(inh_cells, shared_idx)
unk_opto = np.intersect1d(unk_cells, shared_idx)

bins = np.linspace(opto_mod.min(), opto_mod.max(), 30)
ax_opto_exc.hist(opto_mod.loc[exc_opto], bins=bins, color='royalblue', alpha=0.7)
ax_opto_exc.axvline(0, color='k', lw=0.8, ls='--')
ax_opto_exc.set_ylabel("Count")
ax_opto_exc.set_title("Exc", color='royalblue', fontsize=8)
ax_opto_exc.tick_params(labelbottom=False)
ax_opto_exc.set_xlim(-1, 1)

ax_opto_inh.hist(opto_mod.loc[inh_opto], bins=bins, color='tomato', alpha=0.7)
ax_opto_inh.axvline(0, color='k', lw=0.8, ls='--')
ax_opto_inh.set_xlabel("Opto modulation")
ax_opto_inh.set_ylabel("Count")
ax_opto_inh.set_title("Inh", color='tomato', fontsize=8)
ax_opto_inh.set_xlim(-1, 1)

unk_in_mwf = np.intersect1d(unk_cells, mwf.columns)
ax_waveform.plot(mwf[unk_in_mwf].mean(axis=1), color='grey', label='unk')
ax_waveform.fill_between(mwf.index.values, mwf[unk_in_mwf].mean(axis=1) - mwf[unk_in_mwf].sem(axis=1), mwf[unk_in_mwf].mean(axis=1) + mwf[unk_in_mwf].sem(axis=1), color='grey', alpha=0.3)
ax_waveform.plot(mwf[exc_cells].mean(axis=1), color='royalblue', label='exc')
ax_waveform.fill_between(mwf.index.values, mwf[exc_cells].mean(axis=1) - mwf[exc_cells].sem(axis=1), mwf[exc_cells].mean(axis=1) + mwf[exc_cells].sem(axis=1), color='royalblue', alpha=0.3)
ax_waveform.plot(mwf[inh_cells].mean(axis=1), color='tomato', label='inh')
ax_waveform.fill_between(mwf.index.values, mwf[inh_cells].mean(axis=1) - mwf[inh_cells].sem(axis=1), mwf[inh_cells].mean(axis=1) + mwf[inh_cells].sem(axis=1), color='tomato', alpha=0.3)
ax_waveform.set_xlabel("Time (ms)")
ax_waveform.set_xticks([0, 10, 20, 30], np.round(np.array([0, 10, 20, 30])/30, 2))
ax_waveform.set_ylabel("Normalized amplitude")
ax_waveform.legend()

savefig(os.path.expanduser("~/Dropbox/LMNphysio/data/summary_psb_waveforms.png"), dpi=200, facecolor='white')




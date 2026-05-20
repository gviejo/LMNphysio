import _pickle as cPickle
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

data = cPickle.load(open(os.path.expanduser("~/Dropbox/LMNphysio/data/CC_LMN-PSB.pickle"), 'rb'))
alltc = data['alltc']
negpeak = data['negpeak']
zcc = data['zcc']
allcc = data['allcc']

fig = plt.figure(figsize=(10, 5))
gs_outer = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

colors = ['#e07b39', '#5e81ac']

for i, p in enumerate(negpeak.index):
    row, col = i // 3, i % 3
    gs_inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[row, col], wspace=0.5)

    # Left: normalized (z-scored) cross-correlation
    ax_cc = fig.add_subplot(gs_inner[0])
    # cc = zcc[p].loc[-0.02:0.02]
    tmp = allcc['sws'][p].loc[-0.01:0.01]  # use raw cc for plotting, but zoom in on ±20ms
    x = tmp.index.values
    dt = np.mean(np.diff(x))
    x = np.hstack((x - dt / 2, np.array([x[-1] + dt / 2])))

    ax_cc.stairs(tmp.values, x, fill=True, color='k')

    ax_cc.axhline(0, color='gray', lw=0.5, ls='-')
    ax_cc.axvline(0, color='gray', lw=0.5, ls='--')
    ax_cc.set_xlabel("Time (ms)", fontsize=7)
    ax_cc.set_ylabel("Corr.", fontsize=7)
    ax_cc.tick_params(labelsize=6)
    ax_cc.set_title(f"{p[0].split('_')[-1]} — {p[1].split('_')[-1]}", fontsize=7)

    # Right: tuning curves of both neurons in the pair
    ax_tc = fig.add_subplot(gs_inner[1], projection='polar')
    tc0 = alltc[p[0]]
    tc1 = alltc[p[1]]
    ax_tc.plot(tc0.index, tc0.values/tc0.max(), color=colors[0], lw=1.2, label="PSB")
    ax_tc.plot(tc1.index, tc1.values/tc1.max(), color=colors[1], lw=1.2, label="LMN")
    ax_tc.set_xlabel("Angle (°)", fontsize=7)
    ax_tc.set_xticks([])
    ax_tc.set_yticks([])
    # ax_tc.set_ylabel("Rate (Hz)", fontsize=7)
    # ax_tc.tick_params(labelsize=6)
    ax_tc.legend(fontsize=6, bbox_to_anchor=(0.5, 1.8), loc='upper left')

plt.suptitle("PSB–LMN pairs with negative cross-correlation peak during non-REM sleep", fontsize=9, y=0.98)
plt.tight_layout()
plt.savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/data/neg_peak_lmn_psb.png",
    dpi=200,
    facecolor="white",
)
# plt.show()
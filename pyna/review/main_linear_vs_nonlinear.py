import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.ndimage import gaussian_filter
import _pickle as cPickle

dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"

pisi = {'adn':cPickle.load(open(os.path.join(dropbox_path, 'PISI_ADN.pickle'), 'rb')),
        'lmn':cPickle.load(open(os.path.join(dropbox_path, 'PISI_LMN.pickle'), 'rb'))}

pisihd = {}
minmax = []
iocurve = {}
for i, st in enumerate(['adn', 'lmn']):
    pisihd[st] = {}
    iocurve[st] = {}
    for j, e in enumerate(['wak', 'sws']):
        cut = -40
        bins = pisi[st]['bins'][0:cut]
        xt = [np.argmin(np.abs(bins - x)) for x in [10**-2, 1]]
        tmp = pisi[st][e].mean(0)[0:cut]
        tmp2 = np.hstack((tmp, tmp, tmp))
        tmp2 = gaussian_filter(tmp2, sigma=(1, 1))
        tmp3 = tmp2[:,tmp.shape[1]:tmp.shape[1]*2]
        tmp3 = tmp3/tmp3.sum(0)
        tmp3 = tmp3*100.0
        pisihd[st][e] = tmp3
        minmax.append([np.min(tmp3), np.max(tmp3)])
        #
        # tmp4 = pisi[st][e][:,0:cut,:]
        # # tmp5 = gaussian_filter(tmp4, sigma=(0, 1, 1))
        # iocurve[st][e] = np.argmax(tmp4, axis=1)
        # iocurve[st][e] = gaussian_filter(iocurve[st][e], sigma=(0, 0.5))
        # iocurve[st][e] = iocurve[st][e].mean(0)
        iocurve[st][e] = np.argmax(tmp3, axis=0)
        iocurve[st][e] = gaussian_filter(iocurve[st][e], sigma=0.5)
minmax = np.array(minmax)



plt.figure()
gs = plt.GridSpec(3, 2, wspace=0.3, hspace=0.3)
for i, st in enumerate(['adn', 'lmn']):
    for j, e in enumerate(['wak', 'sws']):
        ax = plt.subplot(gs[i, j])
        im = ax.imshow(pisihd[st][e],
                       aspect='auto', origin='lower',
                       extent=(-180, 180, 0, pisihd[st][e].shape[0]),
                       )
        ax.set_title(f"{st.upper()} - {e.upper()}")
        ax.set_xlabel("Centered HD")
        ax.set_ylabel("ISI (s)")
        plt.colorbar(im, ax=ax, label="% of ISIs")

        plt.plot(
            np.linspace(-180, 180, pisihd[st][e].shape[1]),
                iocurve[st][e],
            color='white', linewidth=2)

for j, e in enumerate(['wak', 'sws']):
    ax = plt.subplot(gs[2, j])
    for i, st in enumerate(['adn', 'lmn']):
        ax.plot(
            np.linspace(-180, 180, pisihd[st][e].shape[1]),
                iocurve[st][e],
            label=st.upper(), linewidth=2)
    ax.set_title(f"{e.upper()}")
    ax.set_xlabel("Centered HD")
    ax.set_ylabel("ISI (s)")
    ax.legend()

plt.tight_layout()
plt.savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/data/ISI_func_lmn_adn.png"),
    dpi=300, bbox_inches='tight'
)


#
#
# x = np.linspace(0, 1, 500)
#
# # LMN: linear ramp capped at 1
# lmn = np.minimum(x * 2, 1)
#
# # ADN: sigmoid centered at 0.5
# k = 12  # steepness
# adn = 1 / (1 + np.exp(-k * (x - 0.5)))
# # rescale to [0, 1]
# adn = (adn - adn.min()) / (adn.max() - adn.min())
#
# fig, ax = plt.subplots(figsize=(5, 4))
#
# ax.plot(x, lmn, color="#3B8BD4", linewidth=2.5, label="LMN — linear")
# ax.plot(x, adn, color="#1D9E75", linewidth=2.5, label="ADN — sigmoid")
#
# ax.set_xlabel("Input (distance from preferred direction)", fontsize=12)
# ax.set_ylabel("Normalised output", fontsize=12)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_xticks([0, 1])
# ax.set_xticklabels(["0", "max"])
# ax.set_yticks([0, 0.5, 1])
# ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
# ax.legend(frameon=True, fontsize=11)
# ax.spines[["top", "right"]].set_visible(False)
#
# plt.tight_layout()
# plt.savefig(
#     os.path.expanduser("~/Dropbox/LMNphysio/data/lmn_adn_io_functions.png"),
#     dpi=300, bbox_inches='tight'
# )

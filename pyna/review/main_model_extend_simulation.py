import os
import _pickle as cPickle
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter
from matplotlib import gridspec

filepath = os.path.join(os.path.expanduser("~") + "/Dropbox/LMNphysio/model/model_rings.pickle")

data_model = cPickle.load(open(filepath, 'rb'))


gs_activity = gridspec.GridSpec(
    3, 3, wspace=0.2, hspace=0.4
)


durations = [10000, 10000, 10000]
offsets = [0, 0, 0]

titles = ["'Wake'", "'Sleep'", "'Opto.'"]

for i, st in enumerate(['adn', 'lmn']):
    for j, e in enumerate(['wak', 'sws', 'opto']):
        subplot(gs_activity[i,j])
        # simpleaxis(gca())

        # start = int(sl.start + (sl.stop - sl.start)/2)

        # tmp = data[st][start:start+durations[j]].T
        print(data_model[st][e].shape)
        tmp = data_model[st][e][offsets[j]:offsets[j]+durations[j]]
        tmp = gaussian_filter(tmp, sigma=(2, 2), mode='reflect')
        tmp = tmp / tmp.max()

        # im=imshow(tmp.T, origin='lower', aspect='auto', cmap=cmaps[st])
        im=imshow(tmp.T, origin='lower', aspect='auto', cmap='turbo', vmax=1, vmin=0)

        if j == 0:
            ylabel(st.upper(), rotation=0, labelpad=6, y=0.5)
            yticks([0, tmp.shape[1]])
        else:
            yticks([0, tmp.shape[1]], ["", ""])



        if i == 0:
            title(titles[j], y=0.9)
            xticks([0, durations[j]], ["", ""])

        if i == 1 and j == 1:
            xlabel("Simulation steps")

        if i == 1:
            xticks([0, durations[j]], ["", durations[j]])

        if j == 2:
            # Colorbar
            axip = gca().inset_axes([1.35, 0.0, 0.1, 0.6])
            # noaxis(axip)
            cbar = colorbar(im, cax=axip)
            axip.set_title("Rate", y=1.0)
            axip.set_yticks([0, 1])

show()
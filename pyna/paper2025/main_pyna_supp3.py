# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2025-07-19 15:01:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-08-04 14:35:20
import numpy as np
import pandas as pd
import pynapple as nap

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from scipy.stats import zscore
from scipy.stats import mannwhitneyu

# matplotlib.style.use('seaborn-paper')
import matplotlib.image as mpimg

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
# import hsluv

import os
import sys

from scipy.ndimage import gaussian_filter

try:
    from functions import *
except:
    sys.path.append("../")
    from functions import *








def figsize(scale):
    fig_width_pt = 483.69687  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2  # Aesthetic ratio (you could change this)
    # fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_width = 6
    fig_height = fig_width * golden_mean * 0.5 # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    gca().spines['left'].set_position(('outward', 3))
    gca().spines['bottom'].set_position(('outward', 2))

    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)


def noaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)



# font_dir = [os.path.expanduser("~")+'/Dropbox/CosyneData/figures_poster_2022']
# for font in font_manager.findSystemFonts(font_dir):
#     font_manager.fontManager.addfont(font)

fontsize = 6

COLOR = (0.25, 0.25, 0.25)

# rcParams["font.family"] = 'Liberation Sans'
rcParams["font.family"] = 'sans-serif'
# rcParams["font.family"] = 'DejaVu Sans'
rcParams["font.size"] = fontsize
rcParams["text.color"] = COLOR
rcParams["axes.labelcolor"] = COLOR
rcParams["axes.labelsize"] = fontsize
rcParams["axes.labelpad"] = 1
# rcParams['axes.labelweight'] = 'bold'
rcParams["axes.titlesize"] = fontsize
rcParams["xtick.labelsize"] = fontsize
rcParams["ytick.labelsize"] = fontsize
rcParams["legend.fontsize"] = fontsize
rcParams["figure.titlesize"] = fontsize
rcParams["xtick.major.size"] = 1.3
rcParams["ytick.major.size"] = 1.3
rcParams["xtick.major.width"] = 0.4
rcParams["ytick.major.width"] = 0.4
rcParams["axes.linewidth"] = 0.4
rcParams["axes.edgecolor"] = COLOR
rcParams["axes.axisbelow"] = True
rcParams["xtick.color"] = COLOR
rcParams["ytick.color"] = COLOR
rcParams['xtick.major.pad'] = 0.5
rcParams['ytick.major.pad'] = 0.5
rcParams['xtick.minor.pad'] = 0.5
rcParams['ytick.minor.pad'] = 0.5


colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}


short_epochs = {'wak':'Wake', 'sws':'non-REM', 'rem':'REM'}

cmap = plt.get_cmap("Set2")

###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"
data = cPickle.load(open(os.path.join(dropbox_path, 'All_TC.pickle'), 'rb'))

alltc = data['alltc']
allinfo = data['allinfo']



fig = figure(figsize=figsize(1))

gs3 = GridSpec(1, 3, hspace = 0.3, wspace=0.5)#, height_ratios=[0.2, 0.5, 0.05])




###############
# Random sampling
###############
# gs3 = gridspec.GridSpecFromSubplotSpec(1,4, outergs[0,0],
#     wspace=0.3, #width_ratios=[0.3, 0.25, 0.25, 0.1]
#      )

data_rand = cPickle.load(open(os.path.join(dropbox_path, 'random_LMN_comparison.pickle'), 'rb'))
all_center_tc_real = data_rand['all_center_tc_real']
all_center_tc_fake = data_rand['all_center_tc_fake']
allr_fake = data_rand['allr']

data_real = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
allr_real = data_real['allr'].dropna()

# gs_rand = gridspec.GridSpecFromSubplotSpec(2, 2, gs3[0, 0], hspace=0.5)

# Panel 1: mean centered tuning curves real vs Poisson
ax = subplot(gs3[0, 0])
simpleaxis(ax)
n_bins = all_center_tc_real.shape[0]
x = np.linspace(-180, 180, n_bins)
m_real = np.nanmean(all_center_tc_real, 1)
s_real = np.nanstd(all_center_tc_real, 1) / np.sqrt(np.sum(np.isfinite(all_center_tc_real), 1))
m_fake = np.nanmean(all_center_tc_fake, 1)
s_fake = np.nanstd(all_center_tc_fake, 1) / np.sqrt(np.sum(np.isfinite(all_center_tc_fake), 1))
plot(x, m_real, color=colors['lmn'], lw=1, label='Real')
fill_between(x, m_real - s_real, m_real + s_real, color=colors['lmn'], alpha=0.2)
plot(x, m_fake, color='r', lw=1, label='Sampled')
fill_between(x, m_fake - s_fake, m_fake + s_fake, color='orange', alpha=0.2)
xlabel('Angle from peak (°)')
ylabel('Rate (Hz)')
title(f'non-REM TC (n={all_center_tc_real.shape[1]})')
legend(frameon=False, fontsize=fontsize-1, bbox_to_anchor=(1.0, 0.87), loc='center')
xticks([-180, 0, 180])

# Panel 2: Poisson LMN fake correlation wak vs sws
ax = subplot(gs3[0, 1])
simpleaxis(ax)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
gca().set_aspect('equal')
scatter(allr_fake['wak'], allr_fake['sws'], 0.4, color='orange', alpha=0.5)
m, b = np.polyfit(allr_fake['wak'].values, allr_fake['sws'].values, 1)
xfit = np.linspace(allr_fake['wak'].min(), allr_fake['wak'].max(), 5)
plot(xfit, xfit * m + b, color=COLOR, linewidth=0.5)
r_val = scipy.stats.pearsonr(allr_fake['wak'].values, allr_fake['sws'].values)[0]
xlabel('Wake', labelpad=50)
ylabel("non-REM", labelpad=50)
title(f'Sampled - r={r_val:.2f}')
xlim(-1, 1); ylim(-1, 1)
xticks([-1, 1], ['-1', '1']); yticks([-1, 1], ['-1', '1'])

# Panel 3: real LMN correlation wak vs sws
ax = subplot(gs3[0, 2])
simpleaxis(ax)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
gca().set_aspect('equal')
scatter(allr_real['wak'], allr_real['sws'], 0.4, color=colors['lmn'], alpha=0.5)
m, b = np.polyfit(allr_real['wak'].values, allr_real['sws'].values, 1)
xfit = np.linspace(allr_real['wak'].min(), allr_real['wak'].max(), 5)
plot(xfit, xfit * m + b, color=COLOR, linewidth=0.5)
r_val = scipy.stats.pearsonr(allr_real['wak'].values, allr_real['sws'].values)[0]
xlabel('Wake', labelpad=50)
ylabel("non-REM", labelpad=50)
title(f'Real LMN - r={r_val:.2f}')
xlim(-1, 1); ylim(-1, 1)
xticks([-1, 1], ['-1', '1']); yticks([-1, 1], ['-1', '1'])



gs3.update(top=0.9, bottom=0.14, right=0.98, left=0.08)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2025/supp3.pdf",
    dpi=200,
    facecolor="white",
)
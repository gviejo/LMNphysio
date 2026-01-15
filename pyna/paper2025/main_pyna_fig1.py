# %%
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-08-04 09:54:03
import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, rgb_to_hsv

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
import hsluv

import os
import sys

from scipy.ndimage import gaussian_filter

try:
    from functions import *
except:
    sys.path.append("../")
    from functions import *

import colorcet as cc
import seaborn as sns


def figsize(scale):
    fig_width_pt = 483.69687  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2  # Aesthetic ratio (you could change this)
    # fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_width = 6
    fig_height = fig_width * golden_mean * 1.2  # height in inches
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
rcParams["xtick.major.width"] = 0.2
rcParams["ytick.major.width"] = 0.2
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

cmap = plt.get_cmap("Set2")
# colors = {'adn':cmap(0), "lmn":cmap(1), "psb":cmap(2)}


# clrs = ['sandybrown', 'olive']
# clrs = ['#CACC90', '#8BA6A9']

###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"

s = "A5011-201014A"
name = s

data = cPickle.load(open(dropbox_path + "/DATA_FIG_LMN_ADN_{}.pickle".format(s), "rb"))
# decoding = {
#     "wak": nap.Tsd(t=data["wak"].index.values, d=data["wak"].values, time_units="s"),
#     "sws": nap.Tsd(t=data["sws"].index.values, d=data["sws"].values, time_units="s"),
#     "rem": nap.Tsd(t=data["rem"].index.values, d=data["rem"].values, time_units="s"),
# }
angle = data["angle"]
tcurves = data["tcurves"]
peaks = data["peaks"]
spikes = data["spikes"]
tokeep = data["tokeep"]
adn = data["adn"][1:]
lmn = data["lmn"]
waveforms = data['waveforms']
position = data['position']
exs = {"wak": data["ex_wak"], "rem": data["ex_rem"], "sws": data["ex_sws"]}

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
allrlmn = data['allr']
r_lmn = data['pearsonr']

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'rb'))
allradn = data['allr']
r_adn = data['pearsonr']

angs = {'adn': allradn['ang'], 'lmn': allrlmn['ang']}

data = cPickle.load(open(os.path.join(dropbox_path, 'All_CC_LMN.pickle'), 'rb'))
allcc_lmn = data['allcc']

data = cPickle.load(open(os.path.join(dropbox_path, 'All_CC_ADN.pickle'), 'rb'))
allcc_adn = data['allcc']

allcc = {'adn': allcc_adn, 'lmn': allcc_lmn}

for g in allcc.keys():
    for e in allcc[g].keys():
        tmp = allcc[g][e][angs[g].sort_values().index]
        tmp = tmp.apply(gaussian_filter, axis=0, sigma=15)
        tmp = tmp.apply(zscore)
        allcc[g][e] = tmp

allr = {'adn': allradn, 'lmn': allrlmn}
allr_sess = {'adn': r_adn, 'lmn': r_lmn}

fz = {'adn': np.arctanh(allradn[['wak', 'sws']]), 'lmn': np.arctanh(allrlmn[['wak', 'sws']])}
# fz = {'adn':allradn[['wak', 'sws']], 'lmn':allrlmn[['wak', 'sws']]}

zbins = np.linspace(0, 1.5, 30)

angbins = np.linspace(0, np.pi, 4)

# meancc = {}
# for e in ['wak', 'sws']:
#     meancc[e] = {}
#     for i, g in enumerate(['adn', 'lmn']):
#         groups = angs[g].groupby(np.digitize(angs[g], angbins))
#         for j in range(angbins.shape[0]-1):
#             meancc[e][g+'-'+str(j)] = allcc[g][e][groups.groups[j+1]].mean(1)
#     meancc[e] = pd.DataFrame.from_dict(meancc[e])


p = {}
meanp = {}

for i, g in enumerate(fz.keys()):

    groups = fz[g].groupby(np.digitize(angs[g], angbins))

    for j in range(angbins.shape[0] - 1):
        idx = groups.groups[j + 1]
        z = fz[g].loc[idx]

        count = np.histogram(np.abs(z['wak'] - z['sws']), zbins)[0]
        count = count / np.sum(count)

        p[g + '-' + str(j)] = count
        meanp[g + '-' + str(j)] = np.mean(z['wak'] - z['sws'])

p = pd.DataFrame.from_dict(p)
p = p.set_index(pd.Index(zbins[0:-1] + np.diff(zbins) / 2))
# p = p.rolling(10, win_type='gaussian').sum(std=1)
zgroup = p


proj_data = cPickle.load(open(os.path.join(dropbox_path, "projections_KPCA_LMN_ADN_v2.pickle"), "rb"))

proj = proj_data['proj']
colors_proj = proj_data['colors']



###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, hspace=0.4, height_ratios=[0.2, 0.8])

names = {'adn': "ADN", 'lmn': "LMN"}
epochs = {'wak': 'Wakefulness', 'sws': 'Non-REM sleep'}
short_epochs = {'wak': 'Wake', 'sws': 'Sleep'}

gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], width_ratios=[0.3, 0.3, 0.2], wspace=0.3
)

####################################
# Histo
####################################


# subplot(gs_histo[:, 0])
subplot(gs_top[0, 0])

# noaxis(gca())
# img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/papezhdcircuit.png")
# imshow(img, aspect="equal")

box_width, box_height = 0.5, 0.3
y_positions = [1, 2, 3]
x_position = 0

box_colors = [colors[st] for st in ['lmn', 'adn', 'psb']]
ax = gca()
ax.set_xlim(-1.1, 4.8)
ax.set_ylim(y_positions[0] - box_height / 2 - 0.2, y_positions[-1] + box_height / 2 + 0.2)
# ax.set_aspect('equal')
ax.axis('off')

# Draw boxes
for i, y in enumerate(y_positions):
    outerrect = patches.FancyBboxPatch((x_position - 1, y - box_height),
                                       box_width * 2.8, box_height * 1.8,
                                       boxstyle="round,pad=0.05",
                                       edgecolor=COLOR,
                                       facecolor="white", linewidth=0.5, linestyle='--')
    ax.add_patch(outerrect)
    ax.text(x_position - 0.70, y,
            ['Mammilary\nBody', 'Anterior\nThalamus', 'Cortex'][i], ha='center', va='center', fontsize=4)

    rect = patches.FancyBboxPatch((x_position - box_width / 2, y - box_height / 2),
                                  box_width, box_height,
                                  boxstyle="round,pad=0.1",
                                  edgecolor=None,
                                  facecolor=box_colors[i], linewidth=0)
    ax.add_patch(rect)
    ax.text(x_position, y,
            ['LMN', 'ADN', 'PSB'][i], ha='center', va='center')

# Draw reversed vertical arrows using FancyArrow (Box 3 → 2 → 1)
for i in range(2):
    start_y = y_positions[i]
    arrow = FancyArrow(x_position, start_y + 3 * box_height / 4,
                       0, 0.45,
                       width=0.06,
                       head_length=0.1,
                       head_width=0.15,
                       length_includes_head=True,
                       color="gray")
    ax.add_patch(arrow)

# Right-angle arrow from Box 1 → Box 3 using FancyArrowPatch
x = x_position + box_width / 2 - 0.1
arrow = FancyArrowPatch(
    posA=(x + 0.1, y_positions[-1]),  # Right of top box
    posB=(x + 0.1, y_positions[0]),  # Right of bottom box
    connectionstyle="bar,fraction=-0.2",  # Top down with bend
    arrowstyle="->,head_length=1,head_width=1",
    color="gray",
    linewidth=2,
)
ax.add_patch(arrow)

##############################################################
# PICTURES
##############################################################
gs_histo = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_top[0, 0], width_ratios=[0.8, 0.2], wspace=0.1
)


# axip = ax.inset_axes([3, 0.5, 2, 1], transform=ax.transData)
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN_probes.png")
imagebox = OffsetImage(img, zoom=0.05)
ab = AnnotationBbox(imagebox, (2, 1.0), frameon=False)
ab.patch.set_linewidth(0.05)  # Line width in points
ab.patch.set_edgecolor(COLOR)
ax.add_artist(ab)

img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/adn_probes.png")
imagebox = OffsetImage(img, zoom=0.05)
ab = AnnotationBbox(imagebox, (2, 2.75), frameon=False)
ab.patch.set_linewidth(0.05)  # Line width in points
ab.patch.set_edgecolor(COLOR)
ax.add_artist(ab)

# subplot(gs_histo[0, 1])
# noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/CosyneData/histo_adn.png")
imagebox = OffsetImage(img, zoom=0.15)
ab = AnnotationBbox(imagebox, (4, 2.75), frameon=False)
ab.patch.set_linewidth(0.05)  # Line width in points
ab.patch.set_edgecolor(COLOR)
ax.add_artist(ab)

# imshow(img[:, :, 0], aspect="equal", cmap="viridis")
# ylabel("ADN", rotation=0, labelpad=10)
# xticks([])
# yticks([])

# subplot(gs_histo[1, 1])
# noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/CosyneData/histo_lmn.png")
imagebox = OffsetImage(img, zoom=0.15)
ab = AnnotationBbox(imagebox, (4, 1.25), frameon=False)
ab.patch.set_linewidth(0.05)  # Line width in points
ab.patch.set_edgecolor(COLOR)
ax.add_artist(ab)

# imshow(img[:, :, 0], aspect="equal", cmap="viridis")
# ylabel("LMN", rotation=0, labelpad=10)
# xticks([])
# yticks([])


##############################################
# Trajectory (3D version)
##############################################
ax = subplot(gs_top[0, 1], projection='3d')
ep = nap.IntervalSet(position.time_support.start[0] + 60*12, position.time_support.start[0] + 60*13)

# Get position data
pos_restricted = position.restrict(ep)
x = pos_restricted['x'].values
y = pos_restricted['y'].values  # Using y as the vertical dimension
z = pos_restricted['z'].values

# Color based on head direction
cmap = plt.get_cmap('twilight')
angles_normalized = (pos_restricted['ry'].values % (2 * np.pi)) / (2 * np.pi)
RGB = cmap(angles_normalized)[:, :3]
# RGB = np.array([hsluv.hsluv_to_rgb([h, 100, 50]) for h in np.rad2deg(pos_restricted['ry'].values)])

# Plot 3D trajectory
ax.scatter(x, z, np.zeros_like(x), s=0.6, c=RGB, edgecolors='none')

# # Calculate center and radius for the circular boundary (on x-z plane)
center_xz = position[['x', 'z']].mean(0)
radius = np.max(np.sqrt(np.sum(np.power(position[['x', 'z']].values - center_xz, 2), 1))) + 0.06

# # Draw circle on the floor (x-z plane at min y)
theta = np.linspace(0, 2*np.pi, 50)
r = np.linspace(0, radius, 20)
T, R = np.meshgrid(theta, r)

X = center_xz[0] + R * np.cos(T)
Z = center_xz[1] + R * np.sin(T)
Y = np.zeros_like(X)  # Flat at y=0

ax.plot_surface(X, Z, Y, color='gray', alpha=0.3, edgecolor='none')


# Set axis limits
ax.set_xlim(center_xz[0] - radius - 0.01, center_xz[0] + radius + 0.01)
ax.set_xlim(center_xz[1] - radius - 0.01, center_xz[1] + radius + 0.01)
# ax.set_ylim(y.min(), y.max())

# Labels and ticks
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
# ax.set_xticks([0.15])
# ax.set_xticklabels(["30 cm"])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Set viewing angle
ax.view_init(elev=20, azim=25)

# Remove grid
ax.grid(False)

# Optional: remove axis panes for cleaner look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_axis_off()

# Create polar inset - [left, bottom, width, height] in figure coordinates
pos = ax.get_position()

# Create inset at specific location [left, bottom, width, height]
ax_inset = fig.add_axes([pos.x1+0.04, pos.y1, 0.05, 0.05],
                        projection='polar')

# Create smooth gradient
theta = np.linspace(0, 2*np.pi, 360)
r = np.linspace(0.7, 1.0, 2)  # Inner to outer radius
T, R = np.meshgrid(theta, r)

# # Create color array
# colors_ = np.array([[hsluv.hsluv_to_rgb([np.rad2deg(t) % 360, 100, 50])
#                     for t in theta] for _ in r])
# Normalize angles from radians to [0, 1] range
theta_normalized = (theta % (2 * np.pi)) / (2 * np.pi)
# Get RGB colors from the colormap
colors_ = np.array([[cmap(t)[:3] for t in theta_normalized] for _ in r])


# Plot
ax_inset.pcolormesh(T, R, np.zeros_like(T), color=colors_.reshape(-1, 3),
                    shading='auto')

ax_inset.set_ylim(0, 1)
ax_inset.set_yticks([])
ax_inset.set_xticks([])
ax_inset.spines['polar'].set_visible(False)
ax_inset.grid(False)
ax_inset.set_xticks([0, np.pi])
ax_inset.set_xticklabels(['0°', '180°'], fontsize=fontsize-1)
ax_inset.set_title("Head-direction (°)")

##############################################
# TUNInG CURVES
##############################################
gs_tuning = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_top[0, 2], width_ratios=[1, 1], wspace=0.4
)
for i, (g, idx) in enumerate(zip(['adn', 'lmn'], [adn, lmn])):

    gs2 = gridspec.GridSpecFromSubplotSpec(
        len(idx), 1, subplot_spec=gs_tuning[0, i], hspace=0.1
    )

    for j, n in enumerate(idx):

        subplot(gs2[j, 0])
        simpleaxis(gca())
        h = np.rad2deg(tcurves[n].idxmax())
        # color = hsluv.hsluv_to_rgb([h, 100, 50])
        color = colors[g]
        # plot(tcurves[n], color=color, linewidth=0.5)
        fill_between(
            tcurves[n].index.values,
            np.zeros_like(tcurves[n].values),
            tcurves[n].values,
            color=color,
            alpha=0.8,
        )
        if j != len(idx) - 1:
            xticks([])
        else:
            xticks([0, 2 * np.pi], ["0", "360"])
        yticks([])
        xlim(0, 2 * np.pi)
        if j == len(idx) // 2 and i == 0:
            ylabel("Neurons", rotation=90, labelpad=5)

        if j == 0:
            title(names[g])

        if j == len(idx) - 1 and i == 1:
            xlabel("Head Direction (°)", x=-0.3)



# ##############################################
# # BOTTOM
# ##############################################

gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[1, 0], wspace=0.3, width_ratios=[0.8, 0.2]
)

gs_bottom_left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_bottom[0, 0], hspace=0.4, height_ratios=[1, 1]
)

#####################################
# Mouse head + Raster + RINGS Wake
#####################################
gs_wake = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_bottom_left[0, 0], wspace=0.5, width_ratios=[0.5, 1]
)

# Raster wake ######################
gs_raster = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_wake[0, 0],
    height_ratios=[1, 1, 0.6]
)
e = 'wak'
for i, (g, idx) in enumerate(zip(['adn', 'lmn'], [adn, lmn])):
    subplot(gs_raster[i, 0])
    simpleaxis(gca())
    gca().spines["bottom"].set_visible(False)
    gca().spines['left'].set_bounds(0, len(idx) - 1)
    for k, n in enumerate(idx):
        plot(spikes[n].restrict(exs[e]).fillna(k),
             '|',
             markersize=1.8, markeredgewidth=0.6,
             color=colors[g])
    xlim(exs[e].start[0], exs[e].end[0])
    yticks([len(idx) - 1], [len(idx)])
    if i==0: ylabel("Neurons", y=0)  # , rotation=0, y=0.2, labelpad=10)
    xticks([])
    if i == 0:
        title(epochs[e])


# Decoding #####################
tuning_curves = nap.compute_1d_tuning_curves(spikes[adn], angle, 24, minmax=(0, 2 * np.pi),
                                             ep=angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves)

subplot(gs_raster[2, 0])
simpleaxis(gca())
exex = nap.IntervalSet(exs[e].start[0] - 10, exs[e].end[0] + 10)
da, P = nap.decode_1d(tuning_curves, spikes[adn], exex, 0.1)
da = smoothAngle(da, 1)
d = gaussian_filter(P.values, 3)
tmp2 = nap.TsdFrame(t=P.index.values, d=d, time_support=exs[e])

im = imshow(tmp2.values.T, aspect='auto',
            origin='lower',
            cmap='coolwarm',
            extent=(exs[e].start[0], exs[e].end[0], 0, 2 * np.pi),
            vmin=0
            )


tmp = smoothAngle(angle, 3).restrict(exs[e])
iset = np.abs(tmp.derivative() / 120).threshold(1.0, method='below').time_support


for s_, e_ in iset.values:
    plot(tmp.get(s_, e_), linewidth=0.5, color=COLOR)

plot(tmp.get(s_, e_), linewidth=0.75, color=COLOR, label="Actual HD")
legend(
    handlelength=1,
    loc="center",
    bbox_to_anchor=(0.1, -0.5, 0.5, 0.5),
    framealpha=0,
)


yticks([0, 2 * np.pi], [0, 360])
ylabel("Direction (°)", labelpad=3)


gca().spines["bottom"].set_bounds(exs['wak'].end[0] - 3, exs['wak'].end[0])
xticks([exs['wak'].end[0] - 3, exs['wak'].end[0]], ["", ""])
text(exs['wak'].end[0] - 1.5, -2.2, s="3 sec.", va="center", ha="center")
axip = gca().inset_axes([1.03, 0, 0.04, 0.6])
cbar = colorbar(im, cax=axip)
axip.set_title("P", fontsize=fontsize - 1, y=0.8)
axip.set_yticks([0, 0.15], [0, 0.15])





# Rings wake ######################
gs_rings = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_wake[0, 1], wspace=0.3, hspace=0.03
)
for j, g in enumerate(['lmn', 'adn']):
    subplot(gs_rings[0, j], aspect='equal')
    # simpleaxis(gca())
    Y = proj[e][g]

    if Y.shape[0] > 1000:
        rng = np.random.default_rng(5)
        # idx = np.sort(np.random.choice(np.arange(Y.shape[0]), size=2000, replace=False, random_state=42))
        idx = np.sort(rng.choice(np.arange(Y.shape[0]), size=2000, replace=False))
    else:
        idx = np.arange(Y.shape[0])
    x = Y[idx, 0]
    y = Y[idx, 1]
    clr = colors_proj[e][g][idx]

    # # Converting to twilight
    cmap = plt.cm.twilight
    HSV = rgb_to_hsv(clr)
    H = HSV[:, 0]
    RGB = np.array([cmap(h)[:3] for h in H])
    clr = RGB

    scatter(x, y, c=clr, s=1, alpha=1, edgecolors='none')

    ax = gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.spines['left'].set_bounds(ymin, ymin + 0.2 * (ymax - ymin))
    ax.spines['bottom'].set_bounds(xmin, xmin + 0.2 * (xmax - xmin))
    xticks([xmin + 0.1 * (xmax - xmin)], ["PC1"])
    yticks([ymin + 0.1 * (ymax - ymin)], ["PC2"])

    title(names[g])

#####################################
# Mouse head + Raster + RINGS SWS
#####################################
gs_sws = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_bottom_left[1, 0], wspace=0.4, width_ratios=[0.5, 1]
)



# Raster SWS ######################
gs_raster = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_sws[0, 0],
    height_ratios=[1, 1, 0.6]
)
e = 'sws'
for i, (g, idx) in enumerate(zip(['adn', 'lmn'], [adn, lmn])):
    subplot(gs_raster[i, 0])
    simpleaxis(gca())
    gca().spines["bottom"].set_visible(False)
    gca().spines['left'].set_bounds(0, len(idx) - 1)
    for k, n in enumerate(idx):
        plot(spikes[n].restrict(exs[e]).fillna(k),
             '|',
             markersize=1.8, markeredgewidth=0.6,
             color=colors[g])
    xlim(exs[e].start[0], exs[e].end[0])
    yticks([len(idx) - 1], [len(idx)])
    if i == 0: ylabel("Neurons", y=0)  # , rotation=0, y=0.2, labelpad=10)
    xticks([])
    if i == 0: title(epochs[e])

# Decoding #####################
subplot(gs_raster[2, 0])
simpleaxis(gca())

exex = nap.IntervalSet(exs[e].start[0] - 10, exs[e].end[0] + 10)

da, P = nap.decode_1d(tuning_curves, spikes[adn].count(0.005, exex).smooth(0.01, size_factor=10), exex, 0.005)
da = smoothAngle(da, 1)
d = gaussian_filter(P.values, 3)
tmp2 = nap.TsdFrame(t=P.index.values, d=d, time_support=exs[e])

im = imshow(tmp2.values.T, aspect='auto',
            origin='lower',
            cmap='coolwarm',
            extent=(exs[e].start[0], exs[e].end[0], 0, 2 * np.pi),
            vmin=0
            )

H = np.sum(P * np.log(P.values), 1)
H = H - H.min()
H = H / H.max()
a_ex = H.threshold(0.12).time_support.intersect(exs[e])

# da = da.restrict(a_ex)
# clrs = np.array([hsluv.hsluv_to_rgb([h, 100, 50]) for h in np.rad2deg(da.values)])
# scatter(da.index.values, da.values, s=0.75, c=clrs, edgecolors='none', zorder=2)

for s_, e_ in a_ex.values:
    plot(da.get(s_, e_), 'o', markersize=0.5, markerfacecolor=COLOR, markeredgecolor=None, markeredgewidth=0)
plot(da.get(s_, e_), 'o', markersize=0.5, markerfacecolor=COLOR, markeredgecolor=None, markeredgewidth=0,
     label="Decoded HD")
legend(
    handlelength=1,
    loc="center",
    bbox_to_anchor=(0.0, -0.55, 0.5, 0.5),
    framealpha=0,
    markerscale=4
)
yticks([0, 2 * np.pi], [0, 360])
ylabel("Direction (°)", labelpad=3)
gca().spines["bottom"].set_bounds(exs['sws'].end[0] - 0.5, exs['sws'].end[0])
xticks([exs['sws'].end[0] - 0.5, exs['sws'].end[0]], ["", ""])
text(exs['sws'].end[0] - 0.23, -2.2, s="0.5 sec.", va="center", ha="center")
axip = gca().inset_axes([1.03, 0, 0.04, 0.6])
cbar = colorbar(im, cax=axip)
axip.set_title("P", fontsize=fontsize - 1, y=0.8)
axip.set_yticks([0, 0.1], [0, 0.1])


# Rings SWS ######################
gs_rings = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_sws[0, 1], wspace=0.3, hspace=0.03
)
for j, g in enumerate(['lmn', 'adn']):
    subplot(gs_rings[0, j], aspect='equal')
    # simpleaxis(gca())
    Y = proj[e][g]

    if Y.shape[0] > 1000:
        idx = np.sort(np.random.choice(np.arange(Y.shape[0]), size=2000, replace=False))
    else:
        idx = np.arange(Y.shape[0])
    x = Y[idx, 0]
    y = Y[idx, 1]
    clr = colors_proj[e][g][idx]

    # # Converting to twilight
    cmap = plt.cm.twilight
    HSV = rgb_to_hsv(clr)
    H = HSV[:, 0]
    RGB = np.array([cmap(h)[:3] for h in H])
    clr = RGB

    scatter(x, y, c=clr, s=1, alpha=1, edgecolors='none')

    ax = gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.spines['left'].set_bounds(ymin, ymin + 0.2 * (ymax - ymin))
    ax.spines['bottom'].set_bounds(xmin, xmin + 0.2 * (xmax - xmin))
    xticks([xmin + 0.1 * (xmax - xmin)], ["PC1"])
    yticks([ymin + 0.1 * (ymax - ymin)], ["PC2"])






#####################################
# Pairwise correlation
#####################################
gs_bottom_right = gridspec.GridSpecFromSubplotSpec(
    4, 1, subplot_spec=gs_bottom[0, 1], height_ratios=[1, 1, 0.4, 0.6], hspace=0.06
)


gs_corr1 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_bottom_right[0, 0], wspace=0.5,
)

xlims = (min(np.nanmin(allr['adn']['wak']), np.nanmin(allr['lmn']['wak'])),
         max(np.nanmax(allr['adn']['wak']), np.nanmax(allr['lmn']['wak'])))
ylims = (min(np.nanmin(allr['adn']['sws']), np.nanmin(allr['lmn']['sws'])),
         max(np.nanmax(allr['adn']['sws']), np.nanmax(allr['lmn']['sws'])))
minmax = (min(xlims[0], ylims[0]) - 0.05, max(xlims[1], ylims[1]) + 0.05)

for i, g in enumerate(['adn', 'lmn']):
    subplot(gs_corr1[0, i], aspect='equal')
    simpleaxis(gca())
    tmp = allr[g].dropna()

    plot(tmp['wak'], tmp['sws'], 'o', color=colors[g], alpha=0.5, markeredgewidth=0, markersize=1)
    m, b = np.polyfit(tmp['wak'].values, tmp['sws'].values, 1)
    x = np.linspace(tmp['wak'].min(), tmp['wak'].max(), 5)
    plot(x, x * m + b, color=COLOR, linewidth=0.5)
    if i == 0:
        ylabel(short_epochs['sws'] + " corr. (r)")
    if i == 1:
        xlabel(short_epochs['wak'] + " corr. (r)", x=-0.2)


    gca().set_xticks([-0.5, 0., 0.5])
    gca().set_xticklabels(["-0.5", "0", "0.5"], fontsize=fontsize - 1)
    if i == 0:
        gca().set_yticks([-0.5, 0., 0.5])
        gca().set_yticklabels(["-0.5", "0", "0.5"], fontsize=fontsize - 1)
    else:
        gca().set_yticks([-0.5, 0., 0.5])
        gca().set_yticklabels(["", "", ""], fontsize=fontsize - 1)
    xlim(*minmax)
    ylim(*minmax)
    r, _ = scipy.stats.pearsonr(tmp['wak'].values, tmp['sws'].values)
    title(f"r={np.round(r, 2)}", y=0.8)


###############################################################
## VIOLINPLOT
###############################################################


gs_bottom_left = gridspec.GridSpecFromSubplotSpec(
    4, 3, subplot_spec=gs_bottom_right[1, 0], hspace=0.9, height_ratios=[0.003, 0.4, 0.4, 0.003],
    width_ratios=[0.05, 1, 0.05]
)

subplot(gs_bottom_left[1,1])
simpleaxis(gca())

for i, g in enumerate(['adn', 'lmn']):
    tmp = allr_sess[g]['sws']
    plot(np.ones(len(tmp)) * (i + 1) + np.random.randn(len(tmp)) * 0.05, tmp.values, 'o', color=colors[g], markersize=1)
    plot([i + 1 - 0.2, i + 1 + 0.2], [tmp.mean(), tmp.mean()], linewidth=1, color='grey')

xlim(0.5, 3)
gca().spines['bottom'].set_bounds(1, 2)
ylim(0, 1.1)
gca().spines['left'].set_bounds(0, 1.1)

ylabel("Pop. coherence (r)", y=0, labelpad=3)
# xticks([1, 2], [names['adn'], names['lmn']])
xticks([1, 2], ["", ""])
# title("Sessions")


subplot(gs_bottom_left[2, 1])
simpleaxis(gca())
xlim(0.5, 3)
ylim(-0.1, 1)
gca().spines['bottom'].set_bounds(1, 2)
xlabel("minus baseline", labelpad=1)
# if i == 1: gca().spines["left"].set_visible(False)
plot([1, 2.2], [0, 0], linestyle='--', color=COLOR, linewidth=0.2)
plot([2.2], [0], 'o', color=COLOR, markersize=0.5)
tmp = [allr_sess[g]['sws'] for g in ['adn', 'lmn']]

vp = violinplot(tmp, showmeans=False,
                showextrema=False, vert=True, side='high'
                )
for k, p in enumerate(vp['bodies']):
    p.set_color(colors[['adn', 'lmn'][k]])
    p.set_alpha(1)

m = [a.mean() for a in tmp]
plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

xticks([1, 2], ['ADN', 'LMN'])
# ylabel(r"Mean$\Delta$")


# COmputing tests
map_significance = {
    1: "n.s.",
    2: "*",
    3: "**",
    4: "***"
}

# for i, g in enumerate(['adn', 'lmn']):
#     zw, p = scipy.stats.wilcoxon(pearson[k].values.astype("float"), baseline[k].values.astype("float"), alternative='greater')
#     signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
#     text(i+0.9, m[i]-0.07, s=map_significance[signi], va="center", ha="right")

xl, xr = 2.5, 2.6
plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
zw, p = mannwhitneyu(tmp[1], tmp[0])
print("mannwhitneyu", zw, p, f"n={len(tmp[0])}", f"n={len(tmp[1])}")
signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
text(xr + 0.1, np.mean(m) - 0.07, s=map_significance[signi], va="center", ha="left")




#####################################
# CC LMN -> ADN
#####################################

gs_cc2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_bottom_right[3, 0], wspace=0.5, hspace=0.1, width_ratios=[0.1, 1, 1]
)

data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN-ADN.pickle'), 'rb'))

index = data['zorder'].index.values
zcc = data['zcc']

for i, k in enumerate(['wak', 'sws']):
    subplot(gs_cc2[0, i + 1])
    simpleaxis(gca())

    m = zcc[k][index].mean(1).loc[-0.02:0.02]
    s = zcc[k][index].std(1).loc[-0.02:0.02]

    plot(m, color=cmap(i), linewidth=1)
    fill_between(m.index.values, m.values - s.values, m.values + s.values,
                 alpha=0.2, color=cmap(i), linewidth=0)
    axvline(0, linewidth=0.4, color=COLOR)

    title(short_epochs[k], y=0.9)

    xticks([-0.02, 0, 0.02], [-20, 0, 20])

    if i == 0:
        ylabel("Norm. corr. (Z)")
        text(-0.15, 1.25, "Norm. xcorr LMN-ADN", transform=gca().transAxes)
        # gca().spines['bottom'].set_visible(False)
        # xticks([])
        # yticks([0, 3])
        xlabel("Time lag (ms)", x=1.0)





outergs.update(top=0.96, bottom=0.06, right=0.98, left=0.07)

savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2025/fig1.pdf",
    dpi=200,
    facecolor="white",
)
# show()

# %%

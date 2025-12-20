# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-08-01 10:58:20
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
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrow, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from scipy.stats import zscore

# matplotlib.style.use('seaborn-paper')
import matplotlib.image as mpimg

from pycircstat.descriptive import mean as circmean
import pycircstat.tests as ct

import _pickle as cPickle
# import hsluv

import os
import sys

from scipy.ndimage import gaussian_filter

try:
    from functions import *
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    from functions import *


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
else:
    data_directory = "~"




def figsize(scale):
    fig_width_pt = 483.69687  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2  # Aesthetic ratio (you could change this)
    # fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_width = 6
    fig_height = fig_width * golden_mean * 1.1  # height in inches
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

fontsize = 6.0

COLOR = (0.25, 0.25, 0.25)
cycle = rcParams['axes.prop_cycle'].by_key()['color'][5:]

rcParams["font.family"] = 'sans-serif'
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

opto_color = "#DC143C"

cmap = plt.get_cmap("Set2")
# colors = {'adn':cmap(0), "lmn":cmap(1), "psb":cmap(2)}


# clrs = ['sandybrown', 'olive']
# clrs = ['#CACC90', '#8BA6A9']



# epochs = ['Wake', 'REM sleep', 'nREM sleep']
epochs = ['Wake', 'nREM sleep', 'nREM sleep']


SI_thr = {
    'adn':0.2, 
    'lmn':0.1,
    'psb':0.3
    }    

###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"

filepath = os.path.join(dropbox_path, "DATA_FIG_LMN_PSB_A3019-220701A.pickle")
data = cPickle.load(open(filepath, 'rb'))

tcurves = data['tcurves']
angle = data['angle']
peaks = data['peaks']
spikes = data['spikes']
up_ep = data['up_ep']
down_ep = data['down_ep']
lmn = list(data['lmn'])
psb = list(data['psb'])

exs = {'wak':nap.IntervalSet(start = 9975, end = 9987, time_units='s'),
    'sws':nap.IntervalSet(start = 5800.0, end = 5802.5, time_units = 's')
    }



###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, hspace = 0.3, height_ratios=[0.1, 0.2])


names = {'psb':"PSB", 'lmn':"LMN"}
epochs = {'wak':'Wakefulness', 'sws':'Non-REM sleep'}

gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], width_ratios=[0.6, 0.4, 0.15], wspace=0.4
)


# #####################################
# # Histo
# #####################################
# gs_top_left = gridspec.GridSpecFromSubplotSpec(
#     2, 1, subplot_spec=gs_top[0, 0], hspace=0.5#, height_ratios=[0.5, 0.2, 0.2] 
# )


# subplot(gs_top_left[0,0])
# noaxis(gca())
# img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN-PSB-opto.png")
# imshow(img, aspect="equal")
# xticks([])
# yticks([])

#########################
# Examples LMN PSB raster
#########################
gs_raster_ex = gridspec.GridSpecFromSubplotSpec(
    3, 5, subplot_spec=gs_top[0, 0], width_ratios=[0.12, 0.5, 0.2, 0.5, 0.0], wspace=0.1
)

ms = [0.7, 0.9]
for i, (st, idx) in enumerate(zip(['psb', 'lmn'], [psb, lmn])):
    for j, e in enumerate(['wak', 'sws']):
        subplot(gs_raster_ex[i, [1, 3][j]])
        simpleaxis(gca())
        gca().spines['left'].set_bounds(0, len(idx)-1)
        gca().spines["bottom"].set_visible(False)
        plot(spikes[idx].to_tsd(np.argsort(idx)).restrict(exs[e]), 
            '|', 
            color=colors[st], 
            markersize=ms[i], mew=0.5)

        if j == 0:
            yticks([len(idx)-1], [len(idx)])
            if i == 0:
                ylabel("Neurons", y=0)
        else:
            yticks([])
        xticks([])
        xlim(*exs[e].values)
        if j == 0:
            text(-0.65, 0.5, names[st], transform=gca().transAxes)

        if i == 0:
            title(epochs[e])

        if j == 1:            
            for v in exs[e].intersect(down_ep).values:
                axvspan(v[0], v[1], color='lightgrey', alpha=0.5, linewidth=0)

            if i == 0:
                annotate("Down\nstate", 
                    xy=(v[1], len(psb)-1), 
                    xytext = (v[1]+0.5, len(psb)-3),
                    arrowprops=dict(
                        arrowstyle="-",
                        linewidth=0.1,
                        color=COLOR,
                        shrinkB=0,
                        shrinkA=0.1
                        ),
                    ha="left",
                    va="top",
                    )
            # [axvspan(v[0], v[1], color='lightgrey', alpha=0.5) for v in exs[e].intersect(up_ep).values]

tuning_curves = nap.compute_1d_tuning_curves(spikes[psb], angle, 24, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves)


# Decoding
for i, e in enumerate(['wak', 'sws']):
    subplot(gs_raster_ex[2,[1, 3][i]])
    simpleaxis(gca())
    # if i == 1: gca().spines["left"].set_visible(False)

    exex = nap.IntervalSet(exs[e].start[0] - 10, exs[e].end[0] + 10)
    
    if i == 0:
        da, P = nap.decode_1d(tuning_curves, spikes[psb], exex, 0.1)
    elif i == 1:
        da, P = nap.decode_1d(tuning_curves, spikes[psb].count(0.005, exex).smooth(0.01, size_factor=10), exex, 0.005)
    
    da = smoothAngle(da, 1)

    d = gaussian_filter(P.values, 3)
    tmp2 = nap.TsdFrame(t=P.index.values, d=d, time_support=exs[e])

    # tmp2 = P.restrict(exs[e])

    im = imshow(tmp2.values.T, aspect='auto', 
        origin='lower',
        cmap='coolwarm', 
        extent=(exs[e].start[0], exs[e].end[0], 0, 2*np.pi),
        # vmin=0
        )

    if e == "wak":
        tmp = smoothAngle(angle, 3).restrict(exs['wak'])
        # iset=np.abs(np.gradient(tmp)).threshold(1.0, method='below').time_support
        iset = np.abs(tmp.derivative() / 120).threshold(1.0, method='below').time_support
        for s, e in iset.values:
            plot(tmp.get(s, e), linewidth=0.5, color=COLOR)

        plot(tmp.get(s, e), linewidth=0.75, color=COLOR, label="Actual HD")
        legend(
                handlelength=1,
                loc="center",
                bbox_to_anchor=(0.1, -0.5, 0.5, 0.5),
                framealpha=0,
            )
    elif e == "sws":
        H = np.sum(P*np.log(P.values), 1)
        H = H-H.min()
        H = H/H.max()
        a_ex = H.threshold(0.12).time_support.intersect(exs[e])

        for s, e in a_ex.values:
            plot(da.get(s, e), 'o', markersize= 0.5, markerfacecolor=COLOR, markeredgecolor=None, markeredgewidth=0)
        plot(da.get(s, e), 'o', markersize= 0.5, markerfacecolor=COLOR, markeredgecolor=None, markeredgewidth=0, label="Decoded HD")
        legend(
                handlelength=1,                
                loc="center",
                bbox_to_anchor=(0.0, -0.5, 0.5, 0.5),
                framealpha=0,
                markerscale=4
            )


    if i == 0:
        yticks([0, 2*np.pi], [0, 360])
        ylabel("Direction (°)", labelpad=3)
    else:
        yticks([])

    if i == 0:
        gca().spines["bottom"].set_bounds(exs['wak'].end[0] - 3, exs['wak'].end[0])
        xticks([exs['wak'].end[0] - 3, exs['wak'].end[0]], ["", ""])
        text(exs['wak'].end[0]-1.5, -2.2, s="3 sec.", va="center", ha="center")
    if i == 1:
        gca().spines["bottom"].set_bounds(exs['sws'].end[0] - 0.5, exs['sws'].end[0])
        xticks([exs['sws'].end[0] - 0.5, exs['sws'].end[0]], ["", ""])
        text(exs['sws'].end[0] - 0.23, -2.2, s="0.5 sec.", va="center", ha="center")

    
    axip = gca().inset_axes([1.03, 0, 0.04, 0.6])
    cbar = colorbar(im, cax=axip)
    axip.set_title("P", fontsize=fontsize-1, y=0.8)
    # if i == 0:
    #     axip.set_yticks([0, 0.4], [0, 0.4])
    if i == 1:
        axip.set_yticks([0.03, 0.05])
    elif i== 0:
        axip.set_yticks([0, 0.1])


#####################################
# LMN PSB Connections
#####################################
gs_con = gridspec.GridSpecFromSubplotSpec(2,1, 
    subplot_spec = gs_top[0,1],  hspace = 0.7, wspace=0.2,
    height_ratios=[0.1, 0.2]
    )

data = cPickle.load(open(os.path.expanduser("~/Dropbox/LMNphysio/data/CC_LMN-PSB.pickle"), 'rb'))
alltc = data['alltc']
angdiff = data['angdiff']
allcc = data['allcc']
pospeak = data['pospeak']
negpeak = data['negpeak']
zcc = data['zcc']
order = data['order']



# Example
exn = [ 
        ('A3019-220630A_14', 'A3019-220630A_85'),
        # ('A3018-220613A_70', 'A3018-220613A_73'),
        # ('A3018-220614B_44', 'A3018-220614B_49')
    ]

p = exn[0]
# for i, p in enumerate(exn):
gs_tc2 = gridspec.GridSpecFromSubplotSpec(1,5, gs_con[0,0],
    width_ratios=[0.1, 0.05, 0.1, 0.2, 0.1], wspace=0.1, hspace=0.05
    )

# TUNING CURVES
for j, name, n in zip([0, 2], ['psb', 'lmn'], np.array(exn[0])):
    subplot(gs_tc2[0, j], projection='polar')
    fill_between(alltc.index.values, np.zeros(len(alltc)), alltc[n].values, color=colors[name])
    if i == 0: title(name.upper())
    xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['', '', '', ''])
    yticks([])
    title(name.upper(), pad=2)

# Arrow
subplot(gs_tc2[0,1])
noaxis(gca())
annotate(
'', xy=(1, 0.5), xytext=(0, 0.5),
arrowprops=dict(arrowstyle='->',lw=0.5, color=COLOR)#, headwidth=5, headlength=7.5)
)
xlim(0, 1)
ylim(0, 1)

# CC
subplot(gs_tc2[0,4])
simpleaxis(gca())
tmp = allcc['sws'][p].loc[-0.02:0.02]
x = tmp.index.values
dt = np.mean(np.diff(x))
x = np.hstack((x-dt/2, np.array([x[-1]+dt/2])))
axvspan(0.002, 0.008, alpha=0.2, linewidth=0)
stairs(tmp.values, x, fill=True, color=COLOR)
xlim(-0.01, 0.01)
# ylim(tmp.values.min()-0.5, tmp.values.max()+0.2)
ylabel("Rate (Hz)", labelpad=4)    
xticks([-0.01, 0, 0.01], [-10, 0, 10])
xlabel("Lag (ms)")
print(tmp.idxmax())



####################################
# Hist PSB connections
###################################
gs_con2 = gridspec.GridSpecFromSubplotSpec(2,2, gs_con[1,0],
    hspace=0.5, wspace=0.95
    )

subplot(gs_con2[0,0])
simpleaxis(gca())

for peak in [pospeak, negpeak]:
    hist_, bin_edges = np.histogram(peak.values, bins = np.linspace(-0.01, 0.01, 50), range = (-0.01, 0.01))
    stairs(hist_, bin_edges, fill=True, color=COLOR, alpha=1)

xticks([])
ylabel("%")
axvspan(0.002, 0.008, alpha=0.2, linewidth=0)
xlim(-0.02, 0.02)
title("$Z_{PSB/LMN} > 3$", pad=4)

subplot(gs_con2[1,0])
simpleaxis(gca())
zcc = zcc.apply(lambda x: gaussian_filter1d(x, sigma=1))
a = zcc[order].loc[0.002:0.008].idxmax().sort_values().index
Z = zcc[a].loc[-0.02:0.02]
im=pcolormesh(Z.index.values, np.arange(Z.shape[1]), Z.values.T, cmap='turbo', vmax=3)
xlim(-0.02, 0.02)
xticks([-0.02, 0.0, 0.02], [-20, 0, 20])
xlabel("Lag (ms)")
ylabel("Lag > 0")


# Colorbar
axip = gca().inset_axes([1.05, 0.0, 0.08, 1])
noaxis(axip)
cbar = colorbar(im, cax=axip)
axip.set_title("z", y=0.75)
axip.set_yticks([0, 3])

####################################
# Angular differences
###################################
subplot(gs_con2[1,1])#, projection='polar')
# gca().set_theta_zero_location('N')
simpleaxis(gca())
num_bins = 32
bins = np.linspace(-np.pi, np.pi, num_bins + 1)

# hist(angdiff[order], bins=bins, histtype="stepfilled", facecolor="dimgray", edgecolor="dimgray")

counts, bin_edges = np.histogram(angdiff[order], bins=bins)
counts = (counts/counts.sum())*100
widths = np.diff(bin_edges)
angles = bin_edges[:-1] + widths / 2  # bin centers
# bars = gca().bar(angles, counts, width=widths, bottom=0.0, align='center', linewidth=0, color=COLOR)
fill_between(angles, counts, 0, step="mid", facecolor=COLOR, edgecolor=COLOR, linewidth=0.1)

# # simpleaxis(gca())
# hist(angdiff[order], bins =np.linspace(0, np.pi, 20), color=COLOR)
xlim(-np.pi, np.pi)
xticks([-np.pi, 0, np.pi], ["-180", "0", "180"])
xlabel("Ang. offset")
ylabel("Prop. (%)")
title(r"$PSB \rightarrow LMN$")

####################################
# CORR LMN_PSB
###################################
data = cPickle.load(open(os.path.join(dropbox_path, 'CORR_LMN-PSB_UP_DOWN.pickle'), 'rb'))
pearson = data['pearson']
frates = data['frates']
baseline = data['baseline']

gs_corr_top = gridspec.GridSpecFromSubplotSpec(2,1, gs_top[0,2], hspace=0.7, wspace=0.2)

subplot(gs_corr_top[0,0])
simpleaxis(gca())

for s in pearson.index:
    plot([1, 2], pearson.loc[s,['down', 'decimated']], '-', color=COLOR, linewidth=0.1)
plot(np.ones(len(pearson)), pearson['down'], 'o', color=colors['lmn'], markersize=1)
plot(np.ones(len(pearson))*2, pearson['decimated'], 'o', color=colors['lmn'], markersize=1)

xlim(0.5, 3)
gca().spines['bottom'].set_bounds(1, 2)
ymin = np.minimum(pearson[['decimated','down']].min().min(), 0)
ylim(ymin-0.1, 1.0)
gca().spines['left'].set_bounds(ymin-0.1, 1.0)
yticks([0, 0.5,1], [0, 0.5, 1])
ylabel("Pop. coherence (r)", y=-0.2, labelpad=2)
xticks([1, 2], ['Down', 'Up'])
# title("Sessions")


############
subplot(gs_corr_top[1,0])
simpleaxis(gca())
xlim(0.5, 3)
ylim(-0.1, 1)
gca().spines['bottom'].set_bounds(1, 2)
xlabel("minus baseline", labelpad=1)
# if i == 1: gca().spines["left"].set_visible(False)
plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
plot([2.2], [0], 'o', color=COLOR, markersize=0.5)
tmp = (pearson[['down', 'decimated']]-baseline[['down', 'decimated']]).values.astype("float")
vp = violinplot(tmp, showmeans=False, 
    showextrema=False, vert=True, side='high'
    )
for k, p in enumerate(vp['bodies']): 
    p.set_color(colors['lmn'])    
    p.set_alpha(1)

m = [pearson[c].mean() for c in ['down', 'decimated']]
plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

xticks([1,2],['',''])
# ylabel(r"Mean$\Delta$")

# COmputing tests
map_significance = {
    1:"n.s.",
    2:"*",
    3:"**",
    4:"***"
}

for i, k in enumerate(['down', 'decimated']):
    zw, p = scipy.stats.wilcoxon(pearson[k].values.astype("float"), baseline[k].values.astype("float"), alternative='greater')
    signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    text(i+0.9, m[i]-0.07, s=map_significance[signi], va="center", ha="right")
    print("wilcoxon baseline", zw, p, len(pearson))

xl, xr = 2.5, 2.6
plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
zw, p = scipy.stats.wilcoxon(tmp[:,1], tmp[:,0], alternative='greater')
signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")

print("wilcoxon across", zw, p, len(tmp))









outergs.update(top=0.95, bottom=0.06, right=0.98, left=0.05)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2025/fig2.pdf",
    dpi=200,
    facecolor="white",
)


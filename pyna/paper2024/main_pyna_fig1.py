# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   gviejo
# @Last Modified time: 2025-07-31 21:51:31
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
adn = data["adn"]
lmn = data["lmn"]
waveforms = data['waveforms']

exs = {"wak": data["ex_wak"], "rem": data["ex_rem"], "sws": data["ex_sws"]}


data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
allrlmn = data['allr']
r_lmn = data['pearsonr']

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'rb'))
allradn = data['allr']
r_adn = data['pearsonr']

angs = {'adn':allradn['ang'], 'lmn':allrlmn['ang']}

data = cPickle.load(open(os.path.join(dropbox_path, 'All_CC_LMN.pickle'), 'rb'))
allcc_lmn = data['allcc']

data = cPickle.load(open(os.path.join(dropbox_path, 'All_CC_ADN.pickle'), 'rb'))
allcc_adn = data['allcc']

allcc = {'adn':allcc_adn, 'lmn':allcc_lmn}

for g in allcc.keys():
    for e in allcc[g].keys():
        tmp = allcc[g][e][angs[g].sort_values().index]
        tmp = tmp.apply(gaussian_filter, axis=0, sigma=15)
        tmp = tmp.apply(zscore)
        allcc[g][e] = tmp


allr = {'adn':allradn, 'lmn':allrlmn}
allr_sess = {'adn':r_adn, 'lmn':r_lmn}

fz = {'adn':np.arctanh(allradn[['wak', 'sws']]), 'lmn':np.arctanh(allrlmn[['wak', 'sws']])}
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

    for j in range(angbins.shape[0]-1):
    
        idx = groups.groups[j+1]
        z = fz[g].loc[idx]

        count = np.histogram(np.abs(z['wak'] - z['sws']), zbins)[0]
        count = count/np.sum(count)  

        p[g+'-'+str(j)] = count
        meanp[g+'-'+str(j)] = np.mean(z['wak'] - z['sws'])

p = pd.DataFrame.from_dict(p)
p = p.set_index(pd.Index(zbins[0:-1] + np.diff(zbins)/2))
# p = p.rolling(10, win_type='gaussian').sum(std=1)
zgroup = p


###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(4, 1, hspace = 0.5, height_ratios=[0.4, 0.6, 0.0, 0.25])


names = {'adn':"ADN", 'lmn':"LMN"}
epochs = {'wak':'Wakefulness', 'sws':'Non-REM sleep'}
short_epochs = {'wak':'Wake', 'sws':'Sleep'}

gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[0, 0], width_ratios=[0.5, 0.5], wspace=0.2
)


# gs_top_left = gridspec.GridSpecFromSubplotSpec(
#     2, 1, subplot_spec=gs_top[0, 0], hspace=0.5
# )

#####################################
# Histo
#####################################
# gs_histo = gridspec.GridSpecFromSubplotSpec(
#     2, 2, subplot_spec=gs_top[0, 0], width_ratios=[0.8, 0.2], wspace=0.1
# )

# subplot(gs_histo[:, 0])
subplot(gs_top[0,0])

# noaxis(gca())
# img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/papezhdcircuit.png")
# imshow(img, aspect="equal")

box_width, box_height = 0.5, 0.3
y_positions = [1, 2, 3]
x_position = 0

box_colors = [colors[st] for st in ['lmn', 'adn', 'psb']]
ax = gca()
ax.set_xlim(-1.1, 4.8)
ax.set_ylim(y_positions[0]-box_height/2-0.2, y_positions[-1]+box_height/2+0.2)
# ax.set_aspect('equal')
ax.axis('off')

# Draw boxes
for i, y in enumerate(y_positions):
    outerrect = patches.FancyBboxPatch((x_position - 1, y - box_height),
                                   box_width*2.8, box_height*1.8,
                                   boxstyle="round,pad=0.05",
                                   edgecolor=COLOR,
                                   facecolor="white", linewidth=0.5, linestyle='--')
    ax.add_patch(outerrect)
    ax.text(x_position-0.70, y, 
        ['Mammilary\nBody', 'Anterior\nThalamus', 'Cortex'][i], ha='center', va='center', fontsize=4)

    rect = patches.FancyBboxPatch((x_position - box_width/2, y - box_height/2),
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
    arrow = FancyArrow(x_position, start_y + 3*box_height/4,
                       0, 0.45,
                       width=0.06,
                       head_length=0.1,
                       head_width=0.15,
                       length_includes_head=True,
                       color="gray")
    ax.add_patch(arrow)

# Right-angle arrow from Box 1 → Box 3 using FancyArrowPatch
x = x_position + box_width/2 - 0.1
arrow = FancyArrowPatch(
    posA=(x+0.1, y_positions[-1]),     # Right of top box
    posB=(x+0.1, y_positions[0]),     # Right of bottom box
    connectionstyle="bar,fraction=-0.2",       # Top down with bend
    arrowstyle="->,head_length=1,head_width=1",
    color="gray",
    linewidth=2,
)
ax.add_patch(arrow)



##############################################################
# PICTURES
##############################################################

# axip = ax.inset_axes([3, 0.5, 2, 1], transform=ax.transData)
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN_probes.png")
imagebox = OffsetImage(img, zoom=0.05)
ab = AnnotationBbox(imagebox, (2, 1.0), frameon=False)
ab.patch.set_linewidth(0.05)      # Line width in points
ab.patch.set_edgecolor(COLOR) 
ax.add_artist(ab)



img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/adn_probes.png")
imagebox = OffsetImage(img, zoom=0.05)
ab = AnnotationBbox(imagebox, (2, 2.75), frameon=False)
ab.patch.set_linewidth(0.05)     # Line width in points
ab.patch.set_edgecolor(COLOR) 
ax.add_artist(ab)





# subplot(gs_histo[0, 1])
# noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/CosyneData/histo_adn.png")
imagebox = OffsetImage(img, zoom=0.15)
ab = AnnotationBbox(imagebox, (4, 2.75), frameon=False)
ab.patch.set_linewidth(0.05)      # Line width in points
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
ab.patch.set_linewidth(0.05)      # Line width in points
ab.patch.set_edgecolor(COLOR) 
ax.add_artist(ab)

# imshow(img[:, :, 0], aspect="equal", cmap="viridis")
# ylabel("LMN", rotation=0, labelpad=10)
# xticks([])
# yticks([])

#####################################
# Raster
#####################################
gs_raster = gridspec.GridSpecFromSubplotSpec(
    3, 4, subplot_spec=gs_top[0, 1], width_ratios=[0.5, 0.1, 0.5, 0.04], wspace=0.1
)


# Raster
for i, (g, idx) in enumerate(zip(['adn', 'lmn'], [adn, lmn])):
    for j, e in enumerate(['wak', 'sws']):
        subplot(gs_raster[i, [0, 2][j]])
        simpleaxis(gca())        
        gca().spines["bottom"].set_visible(False)
        gca().spines['left'].set_bounds(0, len(idx)-1)
        # if j == 1: gca().spines["left"].set_visible(False)
        for k,n in enumerate(idx):
            plot(spikes[n].restrict(exs[e]).fillna(k), 
                '|', 
                markersize = 1, markeredgewidth=0.5,
                color=colors[g])
        xlim(exs[e].start[0], exs[e].end[0])
        if j == 0:
            yticks([len(idx)-1], [len(idx)])
            if i == 0:
                ylabel("Neurons", y=0)#, rotation=0, y=0.2, labelpad=10)
        else:
            yticks([])
        xticks([])


        if i == 0:
            title(epochs[e])


tuning_curves = nap.compute_1d_tuning_curves(spikes[adn], angle, 24, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves)


# Decoding
for i, e in enumerate(['wak', 'sws']):
    subplot(gs_raster[2,[0,2][i]])
    simpleaxis(gca())
    # if i == 1: gca().spines["left"].set_visible(False)

    exex = nap.IntervalSet(exs[e].start[0] - 10, exs[e].end[0] + 10)
    
    if i == 0:
        da, P = nap.decode_1d(tuning_curves, spikes[adn], exex, 0.1)
    elif i == 1:
        da, P = nap.decode_1d(tuning_curves, spikes[adn].count(0.005, exex).smooth(0.01, size_factor=10), exex, 0.005)
    
    da = smoothAngle(da, 1)

    d = gaussian_filter(P.values, 3)
    tmp2 = nap.TsdFrame(t=P.index.values, d=d, time_support=exs[e])

    # tmp2 = P.restrict(exs[e])

    im = imshow(tmp2.values.T, aspect='auto', 
        origin='lower',
        cmap='coolwarm', 
        extent=(exs[e].start[0], exs[e].end[0], 0, 2*np.pi),
        vmin=0
        )

    if e == "wak":
        tmp = smoothAngle(angle, 3).restrict(exs['wak'])
        iset=np.abs(np.gradient(tmp)).threshold(1.0, method='below').time_support
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
                bbox_to_anchor=(0.0, -0.55, 0.5, 0.5),
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
        axip.set_yticks([0, 0.1], [0, 0.1])
    elif i== 0:
        axip.set_yticks([0, 0.15], [0, 0.15])



#####################################
# Examples
#####################################
gs_middle = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[1, 0], width_ratios = [0.9, 0.2, 0.15], wspace=0.3
)

gs_top2 = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_middle[0, 0], hspace=0.4, wspace=0.3
)



# adn_idx = [adn[12], adn[13], adn[1]]
adn_idx = [adn[12], adn[13], adn[5]] # 3 6
lmn_idx = [lmn[9], lmn[7], lmn[3]]

for i, (s, idx) in enumerate(zip(['adn', 'lmn'], [adn_idx, lmn_idx])):
    
    # Waveforms + tuning curves
    gs_neuron = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_top2[i, 0], wspace=0.2, hspace=0.2, width_ratios=[1, 1, 0.2]
        )
    xx = [0, 1, 1]
    yy = [0, 0, 1]
    
    for j, n in enumerate(idx):
        gs_ex = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_neuron[yy[j], xx[j]], wspace=0.2, hspace=0.5
            )
                
        subplot(gs_ex[0,0])
        noaxis(gca())
        tmp = waveforms[n]
        tmp = tmp - tmp.mean(0)
        for k in range(tmp.shape[1]):
            plot(tmp[:,k]*2-k*1000, color = colors[s], linewidth=0.5)

        # Tuning curves
        subplot(gs_ex[0,1], projection='polar')
        fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n]),tcurves[n].values,color=colors[s])
        xticks([])
        yticks([])


    tmp = allr[s].dropna()
    idx_ = [name+"_"+str(j) for j in idx]
    p1 = tmp.loc[(idx_[0],idx_[1]),['wak','sws']].values
    p2 = tmp.loc[tuple(np.sort((idx_[0],idx_[2]))),['wak','sws']].values
    
    # Spike counts
    bin_sizes = [0.1, 0.01]
    for j, e in enumerate(['wak', 'sws']):
        gs_count = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_top2[i,1+j], wspace=0.01, hspace=0.4, width_ratios=[0.6,0.4]
            )
        
        
        for k, p in enumerate([(idx[0], idx[1]), (idx[0], idx[2])]):
            subplot(gs_count[k,0])
            simpleaxis(gca())
            rates = []
            for u, n in enumerate(p):
                tmp = spikes[n].count(bin_sizes[j]).smooth(bin_sizes[j]*2).restrict(exs[e])
                # tmp = tmp.smooth(5/tmp.rate)
                tmp = tmp/tmp.max()
                rates.append(tmp)
            # maxr = np.max(rates[0])
            # rates = [a/maxr for a in rates]
            fill_between(rates[0].t, 0, rates[0].d, color=colors[s], alpha=1)
            step(rates[0].t, rates[0].d, linewidth=0.1, color=colors[s], where='mid')

            fill_between(rates[1].t, 0, -rates[1].d, color=colors[s], alpha=1)
            step(rates[1].t, -rates[1].d, linewidth=0.1, color=colors[s], where='mid')

            axhline(0, linewidth=0.1, color=COLOR)
            xlim(exs[e].start[0], exs[e].end[0])
            yticks([-1, 0, 1], [1, 0, 1])
            # gca().set_yticks([])
            # gca().spines["left"].set_visible(False)

            x_o = 1.05
            if e == "wak":
                if k == 0:                    
                    # gca().text(x_o, 0.5, s="$r_{P_1}^{W}$="+f"{np.round(p1[0], 2)}", va="center", ha="left", transform=gca().transAxes, fontsize=fontsize-1)
                    gca().text(x_o, 0.5, s="r="+f"{np.round(p1[0], 2)}", va="center", ha="left", transform=gca().transAxes, fontsize=fontsize-1)
                else:
                    gca().text(x_o, 0.5, s="r="+f"{np.round(p2[0], 2)}", va="center", ha="left", transform=gca().transAxes, fontsize=fontsize-1)
            else:
                if k == 0:
                    gca().text(x_o, 0.5, s="r="+f"{np.round(p1[1], 2)}", va="center", ha="left", transform=gca().transAxes, fontsize=fontsize-1)
                else:
                    gca().text(x_o, 0.5, s="r="+f"{np.round(p2[1], 2)}", va="center", ha="left", transform=gca().transAxes, fontsize=fontsize-1)



            if j == 0 and k == 1:
                gca().spines["bottom"].set_bounds(exs['wak'].end[0] - 2, exs['wak'].end[0])
                xticks([exs['wak'].end[0] - 2, exs['wak'].end[0]], ["", ""])                
                text(exs['wak'].end[0] - 1, -2.2, s="2 sec.", va="center", ha="center")

            elif j == 1 and k == 1:
                gca().spines["bottom"].set_bounds(exs['sws'].end[0] - 0.5, exs['sws'].end[0])
                xticks([exs['sws'].end[0] - 0.5, exs['sws'].end[0]], ["", ""])
                text(exs['sws'].end[0] - 0.25, -2.2, s="0.5 sec.", va="center", ha="center")
            else:
                gca().spines["bottom"].set_visible(False)
                gca().set_xticks([])                

            if j == 0 and k == 0:
                ylabel("Norm. rate", y=0)
            # if k == 0 and j==0:
            #     ylabel("Pair\n1", rotation=0, fontsize=5)
            # if k == 1 and j==0:
            #     ylabel("Pair\n2", rotation=0, fontsize=5)
            if i == 0 and k == 0:
                title(epochs[e])



#####################################
# Pairwise correlation
#####################################
gs_corr1 = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_middle[0, 1], hspace=0.2
)

xlims = (min(np.nanmin(allr['adn']['wak']), np.nanmin(allr['lmn']['wak'])), max(np.nanmax(allr['adn']['wak']), np.nanmax(allr['lmn']['wak'])))
ylims = (min(np.nanmin(allr['adn']['sws']), np.nanmin(allr['lmn']['sws'])), max(np.nanmax(allr['adn']['sws']), np.nanmax(allr['lmn']['sws'])))
minmax = (min(xlims[0],ylims[0])-0.05, max(xlims[1],ylims[1])+0.05)

for i, (g, idx) in enumerate(zip(['adn', 'lmn'], [adn_idx, lmn_idx])):
    subplot(gs_corr1[i,0], aspect='equal')
    simpleaxis(gca())
    tmp = allr[g].dropna()

    plot(tmp['wak'], tmp['sws'], 'o', color = colors[g], alpha = 0.5, markeredgewidth=0, markersize=1)
    m, b = np.polyfit(tmp['wak'].values, tmp['sws'].values, 1)
    x = np.linspace(tmp['wak'].min(), tmp['wak'].max(),5)
    plot(x, x*m + b, color=COLOR, linewidth=0.5)    
    ylabel(short_epochs['sws'] + " corr. (r)")
    if i == 1: 
        xlabel(short_epochs['wak'] + " corr. (r)")
        gca().set_xticks([-0.5, 0., 0.5])
        gca().set_yticks([-0.5, 0., 0.5])


    xlim(*minmax)
    ylim(*minmax)
    r, _ = scipy.stats.pearsonr(tmp['wak'].values, tmp['sws'].values)
    title(f"r={np.round(r, 2)}", y=0.8)

    if i == 0: 
        gca().set_xticks([-0.5, 0., 0.5], ['', '', ''])
        gca().set_yticks([-0.5, 0., 0.5])


    # # Annotation
    idx = [name+"_"+str(j) for j in idx]
    
    

    p1 = tmp.loc[(idx[0],idx[1]),['wak','sws']].values
    p2 = tmp.loc[tuple(np.sort((idx[0],idx[2]))),['wak','sws']].values
    
    plot(p1[0], p1[1], 'o', mec=COLOR, mfc="white", alpha=1, markersize=2)
    plot(p2[0], p2[1], 'o', mec=COLOR, mfc="white", alpha=1, markersize=2)

    annotate("P1", xy=p1, xytext=(p1[0]-0.15, p1[1]+0.16), fontsize = 5)
    annotate("P2", xy=p2, xytext=(p2[0]-0.15, p2[1]+0.16), fontsize = 5)

    # print(p1)
    # print(p2)
    # sys.exit()

    
###############################################################
## VIOLINPLOT
###############################################################




gs_bottom_left = gridspec.GridSpecFromSubplotSpec(
    4, 1, subplot_spec=gs_middle[0,2], hspace=0.8, height_ratios=[0.003, 0.4, 0.4, 0.003]
)


subplot(gs_bottom_left[1,0])
simpleaxis(gca())

for i,g in enumerate(['adn', 'lmn']):

    tmp = allr_sess[g]['sws']
    plot(np.ones(len(tmp))*(i+1) + np.random.randn(len(tmp))*0.05, tmp.values,  'o', color = colors[g], markersize = 1)
    plot([i+1-0.2, i+1+0.2], [tmp.mean(), tmp.mean()], linewidth=1, color = 'grey')

xlim(0.5, 3)
gca().spines['bottom'].set_bounds(1, 2)
ylim(0, 1.1)
gca().spines['left'].set_bounds(0, 1.1)

ylabel("Pop. coherence (r)", y=0, labelpad=3)
xticks([1, 2], [names['adn'], names['lmn']])
# title("Sessions")


subplot(gs_bottom_left[2,0])
simpleaxis(gca())
xlim(0.5, 3)
ylim(-0.1, 1)
gca().spines['bottom'].set_bounds(1, 2)
xlabel("minus baseline", labelpad=1)
# if i == 1: gca().spines["left"].set_visible(False)
plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
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

xticks([1,2],['ADN','LMN'])
# ylabel(r"Mean$\Delta$")


# COmputing tests
map_significance = {
    1:"n.s.",
    2:"*",
    3:"**",
    4:"***"
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
text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")



# ##############################################
# # BOTTOM 
# ##############################################

gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[3, 0], wspace=0.2, width_ratios=[0.5, 0.25, 0.25]
)





gs_bottom1 = gridspec.GridSpecFromSubplotSpec(
    2, 4, subplot_spec=gs_bottom[0,0], wspace=0.4, hspace = 0.6, width_ratios=[0.1, 0.06, 0.5, 0.25]
)


for i, g in enumerate(['adn', 'lmn']):


    subplot(gs_bottom1[i, 1])
    gca().invert_yaxis()
    simpleaxis(gca())

    # [axhline(b, linewidth=0.25, color = 'grey') for b in angbins[1:-1]]    
    # count = []
    # for i, g in enumerate(['adn', 'lmn']):
    tmp = allr[g].dropna()
    angdiff = tmp['ang'].sort_values().values    
    plot(np.arange(len(angdiff)), angdiff, '-', color = colors[g], linewidth=2)
    if i == 0:
        title("HD neuron\npair")
        ylabel("Ang. diff. (deg.)", y=0.0)
    yticks([0, np.pi], ["0", "180"])
    ylim(np.pi, 0)
    # count.append(len(angdiff))
    # count = np.array(count)
    xticks([len(angdiff)-1], [len(angdiff)])

    text(-4, 0.5, g.upper(), transform=gca().transAxes)

# #####################################
# # Cross-corrs
# #####################################



gs_cc = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_bottom1[:, 2], hspace = 0.4, wspace = 0.5, width_ratios=[0.5, 0.5, 0.1]
)


vmin = np.min([np.min(allcc[g][e].values) for g in ['adn', 'lmn'] for e in ['wak', 'sws']])
vmax = np.max([np.max(allcc[g][e].values) for g in ['adn', 'lmn'] for e in ['wak', 'sws']])


for k, g in enumerate(['adn', 'lmn']):
    for u, e in enumerate(['wak', 'sws']):
        subplot(gs_cc[k,u])
        simpleaxis(gca())
        if u == 0:
            Z = allcc[g][e].loc[-40:40]
        else:
            Z = allcc[g][e].loc[-1:1]
        im = imshow(Z.values.T, 
            aspect='auto', vmin=vmin, vmax=vmax,
            cmap='bwr'
            )

        if k == 0 and u == 0:        
            text(0.5, 1.9, "Norm. x corr.", transform=gca().transAxes)
        if k == 1 and u == 0:
            xlabel("Time lag (s)", x = 1.1)
            xticks([0, len(Z)//2, len(Z)], [-40, 0, 40])
        if k == 1 and u == 1:
            xticks([0, len(Z)//2, len(Z)], [-1, 0, 1])
        if k == 0:
            xticks([0, len(Z)//2, len(Z)], ["", "", ""])
            title(short_epochs[e])
        # title(names[g], y=0.85)
        yticks([])


axip = gca().inset_axes([1.15, 0.25, 0.12, 1.2])
colorbar(im, cax=axip)
axip.set_title("Z", y=0.9)
axip.set_yticks([-2, 0, 2], ["-2", "0", "2"])


# #####################################
# # CC GLM
# #####################################

data = cPickle.load(open(os.path.join(dropbox_path, 'All_GLM_CC_LMN_ADN.pickle'), 'rb'))
glm_cc = data['cc']
angdiff = data['angdiff']

# meanglmcc = {}
# for e in ['wak', 'sws']:
#     meanglmcc[e] = {}
#     for i, g in enumerate(['adn', 'lmn']):
#         groups = angdiff[g].groupby(np.digitize(angdiff[g], angbins))
#         for j in range(angbins.shape[0]-1):
#             meanglmcc[e][g+'-'+str(j)] = glm_cc[g][e][groups.groups[j+1]].mean(1)
#     meanglmcc[e] = pd.DataFrame.from_dict(meanglmcc[e])


gs_cc = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_bottom1[:, 3], hspace=0.6, wspace = 0.5, width_ratios=[0.5, 0.1]
)

# for i, b in enumerate([0, 2]):

vmin = np.minimum(*[np.min(glm_cc[g]['sws'].values) for g in ['adn', 'lmn']])
vmax = np.maximum(*[np.max(glm_cc[g]['sws'].values) for g in ['adn', 'lmn']])

axis = []

for k, g in enumerate(['adn', 'lmn']):
    subplot(gs_cc[k,0])
    ax = simpleaxis(gca())
    
    tmp = glm_cc[g]['sws'][angdiff[g].sort_values().index].apply(zscore)
    tmp = tmp.apply(gaussian_filter, axis=0, sigma=3)
    

    im = imshow(tmp.values.T, aspect='auto', vmin=vmin, vmax=vmax,
        cmap='bwr'
        )

    yticks([])
    if k == 1:
        xticks([0, len(tmp)//2, len(tmp)], [-0.5, 0, 0.5])    
    else:
        xticks([0, len(tmp)//2, len(tmp)], ["", "", ""])
    
    # title(names[g], y=0.85)

    if k == 1:
        # ylabel(r"$\beta_t$", rotation=0, labelpad=2)
        xlabel("Time lag (s)")

    axis.append(gca())

axip = axis[1].inset_axes([1.15, 0.25, 0.12, 1.2])
colorbar(im, cax=axip)
axip.set_title(r"$\beta_t$", y=0.9)



axip = axis[0].inset_axes([-0.5, 1.0, 2.5, 2.5])
noaxis(axip)
axip.patch.set_alpha(0.0)

u1 = (0.1, 0.25)
u2 = (0.9, 0.25)
p1 = (0.1, 0.5)
ss = (u1[0]+(u2[0]-u1[0])/2, u1[1])

# Draw circle with Σ symbol at y=0.3 level
circle = plt.Circle((u1[0]+(u2[0]-u1[0])/2, u1[1]), 0.1, fill=True, facecolor='white', ec=COLOR, linewidth=0.5)
axip.add_patch(circle)


# Text labels - all at y=0.3 level
axip.text(u1[0], u1[1], "Unit", fontsize=6, ha='center', va='center')
axip.text(u2[0], u2[1], "Unit", fontsize=6, ha='center', va='center')
axip.text(ss[0], ss[1], 'Σ', fontsize=6, ha='center', va='center')
axip.text(p1[0], p1[1], 'Pop.', fontsize=6, ha='center', va='center')


offset_x = 0.09
offset_y = 0.0
arrow = FancyArrowPatch(
    (u1[0] + offset_x, u1[1] + offset_y), (ss[0] - offset_x, ss[1] - offset_y),
    arrowstyle="->",
    color=COLOR,
    linewidth=1,
    # alpha=alphas[i],
    mutation_scale=5,
    zorder=1
    )
axip.add_patch(arrow)

arrow = FancyArrowPatch(
    (ss[0] + offset_x, ss[1] + offset_y), (u2[0] - offset_x, u2[1] - offset_y),
    arrowstyle="->",
    color=COLOR,
    linewidth=1,
    # alpha=alphas[i],
    mutation_scale=5,
    zorder=1
    )
axip.add_patch(arrow)

arrow = FancyArrowPatch(
    (p1[0] + offset_x, p1[1] + offset_y), (ss[0], ss[1] + 0.05),
    arrowstyle="->",
    color=COLOR,
    linewidth=1,
    mutation_scale=5,
    zorder=1,
    connectionstyle="angle,angleA=0,angleB=90"  # Creates L-shaped path
    )
axip.add_patch(arrow)



axip.set_xlim(0, 1)
axip.set_ylim(0, 1)



#####################################
# CC LMN -> ADN
#####################################

gs_cc2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_bottom[0, 1], wspace = 0.5, hspace = 0.1, width_ratios=[0.5, 1, 1]
)


data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN-ADN.pickle'), 'rb'))

index = data['zorder'].index.values
zcc = data['zcc']



for i, k in enumerate(['wak', 'sws']):
    subplot(gs_cc2[0,i+1])
    simpleaxis(gca())

    m = zcc[k][index].mean(1).loc[-0.02:0.02]
    s = zcc[k][index].std(1).loc[-0.02:0.02]

    plot(m, color=cmap(i), linewidth=1)
    fill_between(m.index.values, m.values - s.values, m.values + s.values, 
        alpha=0.2, color=cmap(i), linewidth=0)
    axvline(0, linewidth=0.4, color=COLOR)

    title(short_epochs[k])

    xticks([-0.02, 0, 0.02], [-20, 0, 20])

    if i == 0:
        ylabel("Norm. corr. (Z)")    
        text(0.7, 1.35, "Norm. x corr\nLMN-ADN", transform=gca().transAxes)
        # gca().spines['bottom'].set_visible(False)
        # xticks([])
        # yticks([0, 3])    
        xlabel("Time lag (ms)", x=1.0)
        





# #####################################
# # GLM LMN -> ADN
# #####################################


data = cPickle.load(open(os.path.join(dropbox_path, 'SCORES_GLM_LMN-ADN.pickle'), 'rb'))
scores = data['scores']


gs_scores = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_bottom[0, -1] #, width_ratios=[0.2, 0.9]
)


subplot(gs_scores[0,0])
noaxis(gca())

n = 5
for i in range(n):
    alpha = 1 if i == 2 else 0.1
    plot([0, 1], np.stack((np.arange(n), [i]*n)), '-', linewidth=0.5, color=COLOR, alpha=alpha)

plot(np.ones(n-1), [0, 1, 3, 4], 'o', color=colors['adn'], markersize=2, alpha=0.5)
plot([1], [2], 'o', color=colors['adn'], markersize=2)

plot(np.zeros(n), np.arange(n), 'o', color=colors['lmn'], markersize=2)

xlim(-0.1, 2.5)
title(r"$LMN \rightarrow ADN$", loc='left')
ylim(-0.5, n-0.5)
# text(0, n, "LMN", horizontalalignment='center', verticalalignment='center')
# text(1, n, "ADN", horizontalalignment='center', verticalalignment='center')




subplot(gs_scores[0, 1])
simpleaxis(gca())   

plot([1,2],[0,0], linestyle='--', color='red', linewidth=0.75)
xticks([1, 2], [short_epochs['wak'], short_epochs['sws']])
yticks([0, 0.4], [0, 0.4])
ylabel("pseudo-R2")
title("GLM Scores")
ylim(-0.1, 0.4)
xlim(0.5, 3)
gca().spines['bottom'].set_bounds(1, 2)
text(0.7, 0.0, "Null\nmodel", 
    color = 'red', fontsize = 5, 
    transform=gca().transAxes)#, bbox=dict(facecolor="white", edgecolor="None"))

tmp = [scores[e]['og'].values for e in ['wak', 'sws']]

vp = violinplot(tmp, showmeans=False, 
    showextrema=False, vert=True, side='high'
    )
for k, p in enumerate(vp['bodies']): 
    p.set_color(cmap(k))
    p.set_alpha(1)

m = [a.mean() for a in tmp]
plot([1, 2], m, 'o', markersize=0.5, color=COLOR)


# COmputing tests

for i, g in enumerate(tmp):
    zw, p = scipy.stats.wilcoxon(g)
    print("Wilcoxon", zw, p, f"n={len(g)}")
    signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    text(i+0.9, m[i]-0.02, s=map_significance[signi], va="center", ha="right")

xl, xr = 2.5, 2.6
plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
zw, p = mannwhitneyu(tmp[1], tmp[0])
print("mannwhitneyu", zw, p, f"n={len(tmp[0])}", f"n={len(tmp[1])}")
signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
text(xr+0.1, np.mean(m)-0.02, s=map_significance[signi], va="center", ha="left")





outergs.update(top=0.96, bottom=0.06, right=0.98, left=0.02)


# savefig(
#     os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/fig1.pdf",
#     dpi=200,
#     facecolor="white",
# )
# show()



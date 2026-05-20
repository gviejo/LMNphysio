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
    fig_height = fig_width * golden_mean * 1.7 # height in inches
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

outergs = GridSpec(3, 1, hspace = 0.3, height_ratios=[0.2, 0.5, 0.05])


#########################
# TUNING CURVES SELECTION
#########################

gs1 = gridspec.GridSpecFromSubplotSpec(2,3, outergs[0,0],
    wspace = 0.5, width_ratios=[0.6, 0.5, 0.3], hspace=0.5
    )

for i, st in enumerate(['adn', 'lmn']):
    gs1_1 = gridspec.GridSpecFromSubplotSpec(2,2, gs1[i,0], wspace=0.5, hspace=0.85
        )

    idx = allinfo[allinfo['location'] == st].index.values

    tokeep = allinfo.loc[idx][(allinfo.loc[idx]['rate'] > 1.0) & (allinfo.loc[idx]['SI'] > 0.1) & (allinfo.loc[idx]['tokeep']==1)].index.values


    # Cloud
    ax = subplot(gs1_1[:,0])
    simpleaxis(ax)

    loglog(allinfo.loc[idx, 'rate'], allinfo.loc[idx, 'SI'], '.', markersize=1, color=COLOR)
    loglog(allinfo.loc[tokeep, 'rate'], allinfo.loc[tokeep, 'SI'], 'o', color=colors[st], markersize=1)

    if i == 1:
        xlabel("Rate (Hz)")
        ylabel("HD info. (bits/spk)", labelpad=5, y=1.0)

    axvline(1.0, linewidth=0.5, color=COLOR)
    axhline(0.1, linewidth=0.5, color=COLOR)

    tc = centerTuningCurves_with_peak(alltc[tokeep])
    tc = tc/tc.max()
    title(st.upper(), y=0.9, x=1.14)

    # Tuning curves
    subplot(gs1_1[0,1])
    simpleaxis(gca())
    m = tc.mean(1)
    s = tc.std(1)
    plot(tc.mean(1), linewidth=1, color=colors[st])
    fill_between(m.index.values, m.values-s.values, m.values+s.values, color=colors[st], alpha=0.2)
    xticks([-np.pi, np.pi], [-180, 180])
    xlim(-np.pi, np.pi)

    ax = subplot(gs1_1[1,1])
    tmp = tc.values.T
    im = imshow(tmp, aspect='auto')

    xticks([0, tmp.shape[1]], [-180, 180])
    yticks([0], [tmp.shape[0]])
    if i == 1:
        xlabel("Angle (deg.)", labelpad=-1)

    if i == 1:
        axip = gca().inset_axes([-0.2, -1.4, 0.5, 0.2])
        noaxis(axip)
        cbar = colorbar(im, cax=axip, orientation='horizontal')
        axip.text(1.05, 0.5, "Rate", transform=axip.transAxes,
          va='center', ha='left')
        # axip.set_title("Rate")
        # axip.set_yticks([0, 3])



##############################
# REM EXAMPLE
###############################
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



# Raster SWS ######################
gs_raster = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs1[:, 1],
    height_ratios=[1, 1, 0.6]
)
e = 'rem'
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
    if i == 0: title("REM sleep")

# Decoding #####################
subplot(gs_raster[2, 0])
simpleaxis(gca())

exex = nap.IntervalSet(exs[e].start[0] - 10, exs[e].end[0] + 10)

tuning_curves = nap.compute_1d_tuning_curves(spikes[adn], angle, 24, minmax=(0, 2 * np.pi),
                                             ep=angle.time_support.loc[[0]])
tuning_curves = smoothAngularTuningCurves(tuning_curves)

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
gca().spines["bottom"].set_bounds(exs[e].end[0] - 1, exs[e].end[0])
xticks([exs[e].end[0] - 1, exs[e].end[0]], ["", ""])
text(exs[e].end[0] - 0.5, -2.2, s="1 sec.", va="center", ha="center")
axip = gca().inset_axes([1.03, 0, 0.04, 0.6])
cbar = colorbar(im, cax=axip)
axip.set_title("P", fontsize=fontsize - 1, y=0.8)
axip.set_yticks([0, 0.1], [0, 0.1])



################################
# REM Correlation
################################



gs_bottom_left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs1[:,-1], hspace=0.6, height_ratios=[0.4, 0.4]
)


names = {'adn':"ADN", 'lmn':"LMN"}

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
r_lmn = data['pearsonr']
data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'rb'))
r_adn = data['pearsonr']
allr_sess = {'adn':r_adn, 'lmn':r_lmn}


subplot(gs_bottom_left[0,0])
simpleaxis(gca())

for i,g in enumerate(['adn', 'lmn']):

    tmp = allr_sess[g]['rem']
    plot(np.ones(len(tmp))*(i+1) + np.random.randn(len(tmp))*0.05, tmp.values,  'o', color = colors[g], markersize = 1)
    plot([i+1-0.2, i+1+0.2], [tmp.mean(), tmp.mean()], linewidth=1, color = 'grey')

xlim(0.5, 3)
gca().spines['bottom'].set_bounds(1, 2)
ylim(0, 1.1)
gca().spines['left'].set_bounds(0, 1.1)

ylabel("Pop. coherence (r)", y=0, labelpad=3)
xticks([1, 2], [names['adn'], names['lmn']])
title("Wake vs REM", y=0.99)


subplot(gs_bottom_left[1,0])
simpleaxis(gca())
xlim(0.5, 3)
ylim(-0.1, 1)
gca().spines['bottom'].set_bounds(1, 2)
xlabel("minus baseline", labelpad=1)
# if i == 1: gca().spines["left"].set_visible(False)
plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
plot([2.2], [0], 'o', color=COLOR, markersize=0.5)
tmp = [allr_sess[g]['rem'] for g in ['adn', 'lmn']]

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
signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")

print("REM acrross", zw, p, len(tmp[0]), len(tmp[1]))

################################
# PAIRWISE CORRELATION / ANIMAL
################################
gs2 = gridspec.GridSpecFromSubplotSpec(2,1, outergs[1,0], 
    height_ratios=[0.3, 0.7], hspace=0.2
    )

i = 0
st = 'adn'


data = cPickle.load(open(os.path.join(dropbox_path, f'All_correlation_{st.upper()}.pickle'), 'rb'))
allr = data['allr'] 
sessions = [s.split("/")[-1] for s in data['pearsonr'].index.values]
animals = np.unique([s.split("-")[0] for s in sessions])

gs2_1 = gridspec.GridSpecFromSubplotSpec(2, len(animals), gs2[i,0])

for k, e in enumerate(['sws', 'rem']):

    for j, a in enumerate(animals):

        index = [k for k in allr.index.values if a in k[0]]

        ax = subplot(gs2_1[k,j])            
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)        
        gca().set_aspect("equal")
        r, _ = scipy.stats.pearsonr(allr.loc[index, 'wak'].values, allr.loc[index, e].values)

        scatter(allr.loc[index, 'wak'], allr.loc[index, e],
                0.4,
                color=colors[st]                    
            )
        xlim(-1, 1)
        ylim(-1,1)        

        gca().text(0.3, -0.9, f"r={np.round(r,2)}", fontsize=fontsize-1)

        if j == 0:
            ylabel(short_epochs[e], labelpad=15)

        if a == 'A5043' and e == 'rem':
            xlabel("Wake", labelpad=15)
            
        if k == 0:
            title(a)

        if i == 0 and k == 0 and j == 0:
            xticks([-1, 1], ['-1', '1'])
            yticks([-1, 1], ['-1', '1'])            
        else:
            xticks([-1, 1], ['', ''])
            yticks([-1, 1], ['', ''])


        m, b = np.polyfit(allr.loc[index, 'wak'].values, allr.loc[index, e].values, 1)
        x = np.linspace(allr.loc[index, 'wak'].values.min(), allr.loc[index, 'wak'].values.max(),5)
        plot(x, x*m + b, color=COLOR, linewidth=0.5)

            
i = 1
st = 'lmn'


data = cPickle.load(open(os.path.join(dropbox_path, f'All_correlation_{st.upper()}.pickle'), 'rb'))
allr = data['allr'] 
sessions = [s.split("/")[-1] for s in data['pearsonr'].index.values]
animals = np.unique([s.split("-")[0] for s in sessions])

gs2_1 = gridspec.GridSpecFromSubplotSpec(2, 1, gs2[i,0])


inds = np.indices((2, 8))
zipind = np.stack((inds[0].flat, inds[1].flat)).T

for k, e in enumerate(['sws', 'rem']):

    gs2_2 = gridspec.GridSpecFromSubplotSpec(2, 8, gs2_1[k,0],
        hspace=0.5)

    for j, a in enumerate(animals):

        index = [k for k in allr.index.values if a in k[0]]

        ax = subplot(gs2_2[zipind[j,0], zipind[j,1]])
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)        
        gca().set_aspect("equal")
        scatter(allr.loc[index, 'wak'], allr.loc[index, e],
                0.4,
                color=colors[st]                    
            )
        xlim(-1, 1)
        ylim(-1,1)

        r, _ = scipy.stats.pearsonr(allr.loc[index, 'wak'].values, allr.loc[index, e].values)
        gca().text(0.9, -0.9, f"r={np.round(r,2)}", fontsize=fontsize-1)

        if j == 0:
            ylabel(short_epochs[e], labelpad=15, y=0)
            
        if a == 'A5043' and e == 'rem':
            xlabel("Wake", labelpad=15)

        if k == 0:
            title(a)

        if i == 0 and k == 0 and j == 0:
            xticks([-1, 1], ['-1', '1'])
            yticks([-1, 1], ['-1', '1'])            
        else:
            xticks([-1, 1], ['', ''])
            yticks([-1, 1], ['', ''])

        m, b = np.polyfit(allr.loc[index, 'wak'].values, allr.loc[index, e].values, 1)
        x = np.linspace(allr.loc[index, 'wak'].values.min(), allr.loc[index, 'wak'].values.max(),5)
        plot(x, x*m + b, color=COLOR, linewidth=0.5)


###############
# Random sampling
###############
gs3 = gridspec.GridSpecFromSubplotSpec(1,4, outergs[2,0],
    wspace=0.3, #width_ratios=[0.3, 0.25, 0.25, 0.1]
     )

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
xlabel('Wake', labelpad=10)
ylabel("non-REM", labelpad=20)
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
xlabel('Wake', labelpad=10)
ylabel("non-REM", labelpad=20)
title(f'Real LMN - r={r_val:.2f}')
xlim(-1, 1); ylim(-1, 1)
xticks([-1, 1], ['-1', '1']); yticks([-1, 1], ['-1', '1'])



outergs.update(top=0.97, bottom=0.04, right=0.98, left=0.08)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2025/supp2.pdf",
    dpi=200,
    facecolor="white",
)
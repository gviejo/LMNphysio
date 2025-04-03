# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-01-31 17:30:00
import numpy as np
import pandas as pd
import pynapple as nap

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches

from scipy.stats import zscore

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
    fig_height = fig_width * golden_mean * 1.2  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
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

fontsize = 7

COLOR = (0.25, 0.25, 0.25)

rcParams["font.family"] = 'DejaVu Sans Mono'
rcParams["font.size"] = fontsize
rcParams["text.color"] = COLOR
rcParams["axes.labelcolor"] = COLOR
rcParams["axes.labelsize"] = fontsize
rcParams["axes.labelpad"] = 3
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
decoding = {
    "wak": nap.Tsd(t=data["wak"].index.values, d=data["wak"].values, time_units="s"),
    "sws": nap.Tsd(t=data["sws"].index.values, d=data["sws"].values, time_units="s"),
    "rem": nap.Tsd(t=data["rem"].index.values, d=data["rem"].values, time_units="s"),
}
angle = data["angle"]
tcurves = data["tcurves"]
peaks = data["peaks"]
spikes = data["spikes"]
tokeep = data["tokeep"]
adn = data["adn"]
lmn = data["lmn"]
waveforms = data['waveforms']

exs = {"wak": data["ex_wak"], "rem": data["ex_rem"], "sws": data["ex_sws"]}

p_decoding = {e:data["p_"+e].restrict(exs[e]) for e in exs.keys()}

# exs["wak"] = nap.IntervalSet(start=7968.0, end=7990.14)
# exs["sws"] = nap.IntervalSet(start=12695.73, end=12701.38)

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



###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, hspace = 0.3, height_ratios=[0.1, 0.1])


names = {'adn':"ADN", 'lmn':"LMN"}
epochs = {'wak':'Wake', 'sws':'Sleep'}

gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], width_ratios=[0.3, 0.3, 0.2], wspace=0.1
)


gs_top_left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_top[0, 0], hspace=0.5
)

#####################################
# Histo
#####################################
gs_histo = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_top_left[0, 0]#, height_ratios=[0.5, 0.2, 0.2] 
)

subplot(gs_histo[:, 0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/tmp.png")
imshow(img, aspect="equal")
xticks([])
yticks([])

subplot(gs_histo[0, 1])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/CosyneData/histo_adn.png")
imshow(img[:, :, 0], aspect="equal", cmap="viridis")
title("ADN")
xticks([])
yticks([])

subplot(gs_histo[1, 1])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/CosyneData/histo_lmn.png")
imshow(img[:, :, 0], aspect="equal", cmap="viridis")
title("LMN")
xticks([])
yticks([])

#####################################
# Raster
#####################################
gs_raster = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=gs_top_left[1, 0]#, height_ratios=[0.5, 0.2, 0.2] 
)


# Raster
for i, (g, idx) in enumerate(zip(['adn', 'lmn'], [adn, lmn])):
    for j, e in enumerate(['wak', 'sws']):
        subplot(gs_raster[i, j])
        simpleaxis(gca())
        gca().spines["left"].set_visible(False)
        gca().spines["bottom"].set_visible(False)
        for k,n in enumerate(idx):
            plot(spikes[n].restrict(exs[e]).fillna(k), 
                '|', 
                markersize = 1, markeredgewidth=0.1,
                color=colors[g])

        if j == 0:
            yticks([len(idx)-1], [len(idx)])
            ylabel(names[g], rotation=0, y=0.2, labelpad=10)
        else:
            yticks([])
        xticks([])


        if i == 0:
            title(epochs[e])

# Position
for i, e in enumerate(['wak', 'sws']):
    subplot(gs_raster[2,i])
    simpleaxis(gca())
    # gca().spines["bottom"].set_visible(False)
    im = imshow(p_decoding[e].values.T, aspect='auto', origin='lower',
        cmap='gist_yarg', extent=(exs[e].start[0], exs[e].end[0], 0, 2*np.pi))
    if e == "wak":
        plot(smoothAngle(angle, 5).restrict(exs['wak']), linewidth=0.2)
    
    if i == 0:
        yticks([0, 2*np.pi], [0, 360])
        ylabel("H.D.\n(deg)", rotation=0, y=0.0, labelpad=10)
    else:
        yticks([])
    # xticks([])

    if i == 0:
        gca().spines["bottom"].set_bounds(exs['wak'].end[0] - 3, exs['wak'].end[0])
        xticks([exs['wak'].end[0] - 1.5], ["3 s"])
    if i == 1:
        gca().spines["bottom"].set_bounds(exs['sws'].end[0] - 0.5, exs['sws'].end[0])
        xticks([exs['sws'].end[0] - 0.25], ["0.5 s"])

    
    axip = gca().inset_axes([0.15, -1.1, 0.05, 0.9])
    cbar = colorbar(im, cax=axip)
    axip.set_ylabel("P", rotation=0, labelpad=-25, y=0.75)
    if i == 0:
        axip.set_yticks([0, 0.4], [0, 0.4])
    if i == 1:
        axip.set_yticks([0, 0.1], [0, 0.1])



#####################################
# Examples
#####################################
gs_top2 = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_top[0, 1], hspace=0.4, wspace=0.5
)




adn_idx = [adn[12], adn[13], adn[1]]
lmn_idx = [lmn[9], lmn[7], lmn[3]]

for i, (s, idx) in enumerate(zip(['adn', 'lmn'], [adn_idx, lmn_idx])):
    
    
    # Waveforms + tuning curves
    gs_neuron = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_top2[i, 0], wspace=0.2, hspace=0.2
        )
    xx = [0, 1, 1]
    yy = [0, 0, 1]
    
    for j, n in enumerate(idx):
        gs_ex = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_neuron[yy[j], xx[j]], wspace=0.1, hspace=0.4
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
    
    
    # Spike counts
    bin_sizes = [0.1, 0.01]
    for j, e in enumerate(['wak', 'sws']):
        gs_count = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=gs_top2[i,1+j], wspace=0.01, hspace=0.4
            )
        
        for k, n in enumerate(idx):            
            subplot(gs_count[k,0])
            simpleaxis(gca())
            tmp = spikes[n].count(bin_sizes[j]).smooth(bin_sizes[j]*2).restrict(exs[e])
            tmp = tmp.smooth(5/tmp.rate)

            bar(
                tmp.t,               
                tmp.d,
                tmp.t[1] - tmp.t[0],
                linewidth=0,
                facecolor=colors[s],
                alpha=0.3
                )
            step(tmp.t, tmp.d, linewidth=0.2, color=colors[s], where='post')
            gca().set_yticks([])
            gca().spines["left"].set_visible(False)

            if j == 0 and k == 2:
                gca().spines["bottom"].set_bounds(exs['wak'].end[0] - 5, exs['wak'].end[0])
                xticks([exs['wak'].end[0] - 2.5], ["5 s"])
            elif j == 1 and k == 2:
                gca().spines["bottom"].set_bounds(exs['sws'].end[0] - 1, exs['sws'].end[0])
                xticks([exs['sws'].end[0] - 0.5], ["1 s"])
            else:
                gca().spines["bottom"].set_visible(False)
                gca().set_xticks([])
                gca().set_yticks([])

            if k == 0 and j==0:
                ylabel("Pair\n1", rotation=0, y=-0.9, fontsize=5)
            if k == 1 and j==0:
                ylabel("Pair\n2", rotation=0, y=-0.9, fontsize=5)

#####################################
# Pairwise correlation
#####################################
gs_corr1 = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_top[0, 2], hspace=0.5, width_ratios=[0.3, 0.5]
)

xlims = (min(np.nanmin(allr['adn']['wak']), np.nanmin(allr['lmn']['wak'])), max(np.nanmax(allr['adn']['wak']), np.nanmax(allr['lmn']['wak'])))
ylims = (min(np.nanmin(allr['adn']['sws']), np.nanmin(allr['lmn']['sws'])), max(np.nanmax(allr['adn']['sws']), np.nanmax(allr['lmn']['sws'])))
minmax = (min(xlims[0],ylims[0]), max(xlims[1],ylims[1]))

for i, (g, idx) in enumerate(zip(['adn', 'lmn'], [adn_idx, lmn_idx])):
    subplot(gs_corr1[i,-1], aspect='equal')
    simpleaxis(gca())
    tmp = allr[g].dropna()

    plot(tmp['wak'], tmp['sws'], 'o', color = colors[g], alpha = 0.5, markeredgewidth=0, markersize=1)
    m, b = np.polyfit(tmp['wak'].values, tmp['sws'].values, 1)
    x = np.linspace(tmp['wak'].min(), tmp['wak'].max(),5)
    plot(x, x*m + b, linewidth=0.1)    
    ylabel(epochs['sws'])
    if i == 1: xlabel(epochs['wak'])
    xlim(*minmax)
    ylim(*minmax)
    title(names[g])

    # # Annotation
    idx = [name+"_"+str(j) for j in idx]
    
    # sys.exit()

    p1 = tmp.loc[(idx[0],idx[1]),['wak','sws']].values
    p2 = tmp.loc[(idx[0],idx[2]),['wak','sws']].values
    
    plot(p1[0], p1[1], 'o', mec=colors[g], mfc="white", alpha=1, markersize=2)
    plot(p2[0], p2[1], 'o', mec=colors[g], mfc="white", alpha=1, markersize=2)

    annotate("P1", xy=p1, xytext=p1+0.05, fontsize = 5)
    annotate("P2", xy=p2, xytext=p2+0.05, fontsize = 5)

    # print(p1)
    # print(p2)
    # sys.exit()

    
###############################################################
## BOTTOM
###############################################################

gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[1, 0], wspace=0.3, width_ratios=[0.4, 0.3, 0.3]
)

gs_bottom1 = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_bottom[0,0], hspace=0.8, width_ratios=[0.2, 0.4, 0.2], height_ratios=[0.1, 0.4]
)
subplot(gs_bottom1[0,1])
simpleaxis(gca())
for i,g in enumerate(['lmn', 'adn']):
    tmp = allr_sess[g]['sws']
    plot(tmp.values, np.ones(len(tmp))*i + np.random.randn(len(tmp))*0.05, 'o', color = colors[g], markersize = 1)
    plot([tmp.mean(), tmp.mean()], [i-0.2, i+0.2], linewidth=1, color = 'grey')
ylim(-0.5, 1.5)
xlim(0, 1)
yticks([1, 0], [names['adn'], names['lmn']])
xlabel("Pearson r")


##############################################
# Hist of Fisher Z
##############################################


gs_fisher = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_bottom1[1, :], hspace = 1, wspace=1.0, width_ratios=[0.1, 0.2]
)

gs_ang_diff = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_fisher[:,0], height_ratios=[0.4, 0.5, 0.4]
)


subplot(gs_ang_diff[1,0])
gca().invert_yaxis()
simpleaxis(gca())

[axhline(b, linewidth=0.25, color = 'grey') for b in angbins[1:-1]]    
count = []
for i, g in enumerate(['adn', 'lmn']):
    tmp = allr[g].dropna()
    angdiff = tmp['ang'].sort_values().values    
    plot(np.arange(len(angdiff)), angdiff, '-', color = colors[g], linewidth=2)
    title("HD neuron pair")
    ylabel("Ang.\ndiff.\n(deg.)", rotation=0, labelpad=0, y=0.2)
    yticks([0, np.pi], ["0", "180"])
    ylim(np.pi, 0)
    count.append(len(angdiff))
count = np.array(count)
xticks(count-1, count, rotation=90)


for i, b in enumerate([0, 2]):   
    subplot(gs_fisher[i,1])
    simpleaxis(gca())

    for j, g in enumerate(['adn', 'lmn']):    
        step(p.index.values, p[g+"-"+str(b)]*100, np.mean(np.diff(zbins)), 
            label=names[g], color=colors[g])

        ylabel("%", labelpad=-10)
        # ylim(0, 25)
        yticks([0, 25])
        
    if i == 1:
        xlabel("$|Z_{Wake} - Z_{Sleep}|$")
    # if i == 0: 
    #     legend(
    #         handlelength=0.5,
    #         loc="center",
    #         bbox_to_anchor=(0.5, 0.8, 0.5, 0.5),
    #         framealpha=0,
    #     )        

#####################################
# Cross-corrs
#####################################

gs_bottom2 = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_bottom[0, 1], hspace = 0.4, wspace = 1,
    height_ratios=[0.4, 0.3, 0.4]
)


gs_cc = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_bottom2[0, 0], wspace = 0.8, width_ratios=[0.5, 0.5, 0.1]
)


vmin = np.minimum(*[np.min(allcc[g]['sws'].values) for g in ['adn', 'lmn']])
vmax = np.maximum(*[np.max(allcc[g]['sws'].values) for g in ['adn', 'lmn']])


for k, g in enumerate(['adn', 'lmn']):
    subplot(gs_cc[0,k])
    simpleaxis(gca())
    Z = allcc[g]['sws'].loc[-1:1]
    im = imshow(Z.values.T, 
        aspect='auto', vmin=vmin, vmax=vmax,
        cmap='bwr'
        )

    if k == 0:
        ylabel("Z", rotation=0, labelpad=10)
        xlabel("Time lag (s)", x = 1.5)

    xticks([0, len(Z)//2, len(Z)], [-1, 0, 1])
    title(names[g])
    yticks([])


axip = gca().inset_axes([1.1, 0, 0.07, 1])
colorbar(im, cax=axip)


#####################################
# CC GLM
#####################################

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
    1, 3, subplot_spec=gs_bottom2[2, 0], wspace = 0.8, width_ratios=[0.5, 0.5, 0.1]
)

# for i, b in enumerate([0, 2]):

vmin = np.minimum(*[np.min(glm_cc[g]['sws'].values) for g in ['adn', 'lmn']])
vmax = np.maximum(*[np.max(glm_cc[g]['sws'].values) for g in ['adn', 'lmn']])



for k, g in enumerate(['adn', 'lmn']):
    subplot(gs_cc[0,k])
    simpleaxis(gca())
    
    tmp = glm_cc[g]['sws'][angdiff[g].sort_values().index].apply(zscore)
    tmp = tmp.apply(gaussian_filter, axis=0, sigma=3)
    

    im = imshow(tmp.values.T, aspect='auto', vmin=vmin, vmax=vmax,
        cmap='bwr'
        )

    yticks([])
    xticks([0, len(tmp)//2, len(tmp)], [-0.5, 0, 0.5])    
    
    if k == 0:
        ylabel(r"$\beta_t$", rotation=0, labelpad=10)
        xlabel("Time lag (s)", x=1.5)

axip = gca().inset_axes([1.1, 0, 0.07, 1])
colorbar(im, cax=axip)



# axip = gca().inset_axes([-0.6, 1.25, 1, 0.5])
axip = subplot(gs_bottom2[1,0])
noaxis(axip)
axip.patch.set_alpha(0.0)
axip.annotate('Pop.', xy=(0.7,0.3), xytext=(0.1, 0.58), color = COLOR,
    arrowprops=dict(facecolor='green',
        headwidth=1.5,
        headlength=1,
        width=0.01,
        ec='grey'
        ),
    fontsize = 6
    )
axip.annotate('', xy=(0.7,0.1), xytext=(0.4, 0.1), 
    arrowprops=dict(facecolor=COLOR,
        headwidth=1.5,
        headlength=1,
        width=0.01,
        ec='grey'                
        ),            
    )        
axip.text(0.0, 0.0, "Unit", fontsize = 6)
axip.text(0.8, 0.0, "Unit", fontsize = 6)
axip.text(0.5, -0.25, r"$\beta_t$", fontsize = 6)
axip.set_xlim(-0.5, 1.6)
axip.set_ylim(0, 2)
# axip.text(0.6, 0.5, r"$\beta_t^{P}$", fontsize = 6)



#####################################
# CC LMN -> ADN
#####################################
gs_final = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_bottom[0, 2], wspace = 1, hspace = 1
)

gs_cc2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_final[0, 0], wspace = 0.6, hspace = 1
)


data = cPickle.load(open(os.path.join(dropbox_path, 'CC_LMN-ADN.pickle'), 'rb'))
allcc = data['allcc']

groups = data['angdiff'].groupby(np.digitize(data['angdiff'], angbins)).groups



k = 1
subplot(gs_cc2[0,0])
simpleaxis(gca())
tmp = allcc[e][groups[k]]
tmp = pd.DataFrame(index=tmp.index, data=tmp.values[::-1]).apply(zscore).loc[-0.01:0.01]
plot(tmp.mean(1), '-', color=cmap(4))
# plot(tmp.mean(1).loc[-0.01:0.01], '-', color=cmap(4))
# imshow(tmp.values.T, aspect='auto', cmap='jet')
# m = tmp.mean(1).loc[-0.01:0.01]
# s = tmp.std(1).loc[-0.01:0.01]
# fill_between(m.index.values, m.values-s, m.values+s, color=cmap(4), alpha=0.1)
axvline(0, color = 'grey', linewidth=0.5)
title("CC LMN -> ADN (Sleep)", x=1.5)
xlabel("Time lag (ms)")
# ylabel("Norm.", rotation=0, y=0.8)
xticks([-0.01, 0, 0.01], [-10, 0, 10])
# # gca().add_patch(Rectangle((-0.01, 2), 0.02, 1.4, facecolor="blue"))
# sys.exit()


subplot(gs_cc2[0,1])
simpleaxis(gca())

tmp = allcc['sws'][groups[1]].loc[-0.01:0.01].idxmax()*-1.0
h, b = np.histogram(tmp, np.linspace(-0.01, 0.01, 21))

bar(b[0:-1], h, np.diff(b).mean(), color=cmap(4))
axvline(0, color = 'grey', linewidth=0.5)
xticks([-0.01, 0, 0.01], [-10, 0, 10])

#####################################
# GLM LMN -> ADN
#####################################


data = cPickle.load(open(os.path.join(dropbox_path, 'SCORES_GLM_LMN-ADN.pickle'), 'rb'))
scores = data['scores']


gs_scores = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_final[1, 0], width_ratios=[0.2, 0.9]
)


subplot(gs_scores[0, 1])
simpleaxis(gca())   
x = 0
for i, e in enumerate(['wak', 'sws']):
    # for j, k in enumerate(['og', 'rnd']):
    k = 'og'
    tmp = scores[e][k]
    # plot(np.ones(len(tmp))*x+np.random.randn(len(tmp))*0.1, scores[e][k].values, 'o', markersize=1)
    # plot([x-0.15, x+0.15], [scores[e][k].values.mean()]*2, color = COLOR, linewidth=1)
    plot(scores[e][k].values, np.ones(len(tmp))*x+np.random.randn(len(tmp))*0.1, 'o', markersize=1, color=cmap(i))
    plot([scores[e][k].values.mean()]*2, [x-0.15, x+0.15], color = COLOR, linewidth=1)
    x += 1
    # x += 1

axvline(0, linestyle='--', color='red', linewidth=1)


yticks([0, 1], [epochs['wak'], epochs['sws']])
xlabel("pseudo-R2", rotation=0)
title("GLM LMN -> ADN")

text(-0.02, 2.0, "Null model", color = 'red', fontsize = 5, bbox=dict(facecolor="white", edgecolor="None"))

xlim(-0.05, 0.4)
ylim(-0.5, 2.5)



outergs.update(top=0.96, bottom=0.09, right=0.98, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/fig1.pdf",
    dpi=200,
    facecolor="white",
)
# show()



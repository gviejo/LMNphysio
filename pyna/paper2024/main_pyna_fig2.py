# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   gviejo
# @Last Modified time: 2025-05-11 20:58:30
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

fontsize = 6

COLOR = (0.25, 0.25, 0.25)
cycle = rcParams['axes.prop_cycle'].by_key()['color']

rcParams["font.family"] = 'Liberation Sans'
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
rcParams['xtick.major.pad'] = 0.5
rcParams['ytick.major.pad'] = 0.5
rcParams['xtick.minor.pad'] = 0.5
rcParams['ytick.minor.pad'] = 0.5

colors = {"adn": "#EA9E8D", "lmn": "#8BA6A9", "psb": "#CACC90"}

cmap = plt.get_cmap("Set2")
# colors = {'adn':cmap(0), "lmn":cmap(1), "psb":cmap(2)}


# clrs = ['sandybrown', 'olive']
# clrs = ['#CACC90', '#8BA6A9']

mks = 1.8
alp = 1
medw = 0.9

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
lmn = data['lmn']
psb = data['psb']

exs = {'wak':nap.IntervalSet(start = 9968.5, end = 9987, time_units='s'),
    'sws':nap.IntervalSet(start = 5800.71, end = 5805.2, time_units = 's')
    }



###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, hspace = 0.4, height_ratios=[0.1, 0.2])


names = {'adn':"ADN", 'lmn':"LMN"}
epochs = {'wak':'Wake', 'sws':'Sleep'}

gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outergs[0, 0], width_ratios=[0.4, 0.4, 0.3], wspace=0.5
)


#####################################
# Histo
#####################################
gs_top_left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_top[0, 0], hspace=0.5#, height_ratios=[0.5, 0.2, 0.2] 
)


subplot(gs_top_left[0,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN-PSB-opto.png")
imshow(img, aspect="equal")
xticks([])
yticks([])

#########################
# Examples LMN PSB raster
#########################
gs_raster_ex = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_top_left[1, 0]#, height_ratios=[0.5, 0.2, 0.2] 
)

ms = [0.7, 0.9]
for i, (st, idx) in enumerate(zip(['psb', 'lmn'], [psb, lmn])):
    for j, e in enumerate(['wak', 'sws']):
        subplot(gs_raster_ex[i,j])
        simpleaxis(gca())
        gca().spines["left"].set_visible(False)
        gca().spines["bottom"].set_visible(False)        
        plot(spikes[idx].to_tsd(np.argsort(idx)).restrict(exs[e]), '|', color=colors[st], markersize=ms[i], mew=0.25)

        if j == 0:
            yticks([len(idx)-1], [len(idx)])
            ylabel(st.upper(), rotation=0, y=0.2, labelpad=10)
        else:
            yticks([])
        xticks([])


        if i == 0:
            title(epochs[e])



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
axvspan(0.002, 0.008, alpha=0.2)    
stairs(tmp.values, x, fill=True, color=COLOR)
xlim(-0.01, 0.01)
# ylim(tmp.values.min()-0.5, tmp.values.max()+0.2)
ylabel("Rate (Hz)", labelpad=4)    
xticks([-0.01, 0, 0.01], [-10, 0, 10])
xlabel("Lag (ms)")



####################################
# Hist PSB connections
###################################
gs_con2 = gridspec.GridSpecFromSubplotSpec(2,2, gs_con[1,0],
    hspace=0.1, wspace=0.9
    )

subplot(gs_con2[0,0])
simpleaxis(gca())

for peak in [pospeak, negpeak]:
    hist_, bin_edges = np.histogram(peak.values, bins = np.linspace(-0.01, 0.01, 50), range = (-0.01, 0.01))
    stairs(hist_, bin_edges, fill=True, color=COLOR, alpha=1)
xlabel("Lag (ms)")
xticks([])
ylabel("%")
axvspan(0.002, 0.008, alpha=0.2)
xlim(-0.02, 0.02)
title("$Z_{PSB-LMN} > 3$", pad=4)

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
subplot(gs_con2[:,1])
simpleaxis(gca())
hist(angdiff[order], bins =np.linspace(0, np.pi, 20), color=COLOR)
xlim(0, np.pi)
xticks([0, np.pi], ["0", "180"])
xlabel("Ang. offset")

####################################
# CORR LMN_PSB
###################################
data = cPickle.load(open(os.path.join(dropbox_path, 'CORR_LMN-PSB_UP_DOWN.pickle'), 'rb'))
pearson = data['pearson']
frates = data['frates']
baseline = data['baseline']

gs_corr_top = gridspec.GridSpecFromSubplotSpec(2,2, gs_top[0,2], hspace=0.9, wspace=0.2)

subplot(gs_corr_top[0,0])
simpleaxis(gca())

for s in pearson.index:
    plot([1, 2], pearson.loc[s,['down', 'decimated']], '-', color=COLOR, linewidth=0.1)
plot(np.ones(len(pearson)), pearson['down'], 'o', color=cycle[0], markersize=1)
plot(np.ones(len(pearson))*2, pearson['decimated'], 'o', color=cycle[1], markersize=1)

xlim(0.5, 2.5)
gca().spines['bottom'].set_bounds(1, 2)
ymin = pearson[['decimated','down']].min().min()
ylim(ymin-0.1, 1.1)
gca().spines['left'].set_bounds(ymin-0.1, 1.1)

ylabel("Pearson r")
xticks([1, 2], ['Down\nstate', 'Up\nstate'])
title("Sessions")


############
subplot(gs_corr_top[1,0])
simpleaxis(gca())
xlim(0.5, 2.5)
ylim(-0.1, 1)
gca().spines['bottom'].set_bounds(1, 2)
xlabel("minus baseline")
# if i == 1: gca().spines["left"].set_visible(False)
plot([1,2],[0,0], linestyle='--', color=COLOR, linewidth=0.1)
tmp = (pearson[['decimated', 'down']]-baseline[['decimated', 'down']]).values.astype("float")
vp = violinplot(tmp, showmeans=False, 
    showextrema=False, vert=True, side='high'
    )
for k, p in enumerate(vp['bodies']): p.set_color(cycle[k])

m = [pearson[c].mean() for c in ['decimated', 'down']]
plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

xticks([1,2],['',''])
ylabel(r"Mean$\Delta$")
# if i == 1: 
#     yticks([])
#     gca().spines["left"].set_visible(False)








#####################################
#####################################
# OPTO
#####################################
#####################################


#####################################
# Histo
#####################################
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[1, 0], width_ratios=[0.05, 0.9], wspace=0.3
)

gs_histo = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs_bottom[0, 0], hspace=0.0
)


subplot(gs_histo[0,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN-PSB-opto.png")
imshow(img, aspect="equal")
xticks([])
yticks([])


subplot(gs_histo[1,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/A8066_S7_3_2xMerged.png")
imshow(img, aspect="equal")
xticks([])
yticks([])


subplot(gs_histo[2,0])
noaxis(gca())
img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/A8066_S7_3_4xMerged.png")
imshow(img, aspect="equal")
xticks([])
yticks([])


#####################################
# Examples LMN IPSILATERAL
#####################################
st = 'lmn'

gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, gs_bottom[0,1], hspace=0.5, wspace=0.9)


exs = [
    ("A8000/A8066/A8066-240216A", nap.IntervalSet(6292.5, 6322.3679), "Wakefulness"),
    # ("A8000/A8066/A8066-240216B", nap.IntervalSet(4076.9, 4083.6), "non-REM Sleep")
    # ("A8000/A8066/A8066-240216B", nap.IntervalSet(4033.1, 4037.5), "non-REM sleep")
    ("A8000/A8066/A8066-240216B", nap.IntervalSet(4033.375, 4037.225), "non-REM sleep")
]

for i, (s, ex, name) in enumerate(exs):

    path = os.path.join(data_directory, "OPTO", s)

    spikes, position, wake_ep, opto_ep, sws_ep = load_opto_data(path, st)

    # Decoding 

    
    gs2_ex = gridspec.GridSpecFromSubplotSpec(2, 1, gs2[i,0], hspace = 0.3)
    
    subplot(gs2_ex[0,0])
    simpleaxis(gca())    
    gca().spines['bottom'].set_visible(False)
    ms = [0.7, 0.9]
    plot(spikes.to_tsd("order").restrict(ex), '|', color=colors[st], markersize=ms[i], mew=0.25)

    [axvspan(s, e, alpha=0.2, color="lightsalmon") for s, e in opto_ep.intersect(ex).values]
    # ylim(-2, len(spikes)+2)
    xlim(ex.start[0], ex.end[0])
    xticks([])
    yticks([0, len(spikes)-1], [1, len(spikes)])
    gca().spines['left'].set_bounds(0, len(spikes)-1)
    title(name)
    ylabel("LMN")
    

    exex = nap.IntervalSet(ex.start[0] - 10, ex.end[0] + 10)
    p = spikes.count(0.01, exex).smooth(0.04, size_factor=20)
    d=np.array([p.loc[i] for i in spikes.index[np.argsort(spikes.order)]]).T
    p = nap.TsdFrame(t=p.t, d=d, time_support=p.time_support)
    p = np.sqrt(p / p.max(0))
    # p = 100*(p / p.max(0))


    subplot(gs2_ex[1,0])
    simpleaxis(gca())
    tmp = p.restrict(ex)
    d = gaussian_filter(tmp.values, 2)
    tmp2 = nap.TsdFrame(t=tmp.index.values, d=d)

    im = pcolormesh(tmp2.index.values, 
            np.linspace(0, 2*np.pi, tmp2.shape[1]),
            tmp2.values.T, cmap='GnBu', antialiased=True)

    x = np.linspace(0, 2*np.pi, tmp2.shape[1])
    yticks([0, 2*np.pi], [1, len(spikes)])
    vls = opto_ep.intersect(ex).values[0] 
    gca().spines['bottom'].set_bounds(vls[0], vls[1])        
    xticks(vls, ['', ''])
    if i == 0: xlabel("10 s", labelpad=-1)
    if i == 1: xlabel("1 s", labelpad=-1)

    if name == "Wakefulness":
        tmp = position['ry'].restrict(ex)
        iset=np.abs(np.gradient(tmp)).threshold(1.0, method='below').time_support
        for s, e in iset.values:
            plot(tmp.get(s, e), linewidth=0.5, color=COLOR)

    # Colorbar
    axip = gca().inset_axes([1.05, 0.0, 0.05, 0.75])
    noaxis(axip)
    cbar = colorbar(im, cax=axip)
    axip.set_title("r", y=0.75)
    axip.set_yticks([0.25, 0.75])


##########################################
# LMN OPTO
##########################################

# gs_corr = gridspec.GridSpecFromSubplotSpec(
#     2, 2, subplot_spec=gs_bottom[0, 2], hspace=1.0#, height_ratios=[0.5, 0.2, 0.2] 
# )

orders = [('lmn', 'opto', 'ipsi', 'opto'), 
            # ('lmn', 'opto', 'ipsi', 'sws'), 
            ('lmn', 'ctrl', 'ipsi', 'opto')]

ranges = {
    "OPTO_SLEEP":(-0.9,0,1,1.9),
    "OPTO_WAKE":(-9,0,10,19)
    }



titles = ['Wakefulness', 'non-REM sleep']

for i, f in enumerate(['OPTO_WAKE', 'OPTO_SLEEP']):

    data = cPickle.load(open(os.path.expanduser(f"~/Dropbox/LMNphysio/data/{f}.pickle"), 'rb'))

    allr = data['allr']
    corr = data['corr']
    change_fr = data['change_fr']
    allfr = data['allfr']
    baseline = data['baseline']

    ####################
    # FIRING rate change
    ####################
    gs_corr2 = gridspec.GridSpecFromSubplotSpec(1,2, gs2[i,1], width_ratios=[0.2, 0.1])

    subplot(gs_corr2[0,0])
    simpleaxis(gca())

    keys = orders[0]
    tmp = allfr[keys[0]][keys[1]][keys[2]]
    tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1.5, mode='constant'))
    tmp = tmp.loc[ranges[f][0]:ranges[f][-1]]
    m = tmp.mean(1)
    s = tmp.std(1)
    
    # plot(tmp, color = 'grey', alpha=0.2)
    plot(tmp.mean(1), color = COLOR)
    fill_between(m.index.values, m.values-s.values, m.values+s.values, color=COLOR, alpha=0.2, ec=None)
    axvspan(ranges[f][1], ranges[f][2], alpha=0.2, color='lightsalmon', edgecolor=None)    
    if i == 1: xlabel("Stim. time (s)", labelpad=1)
    xlim(ranges[f][0], ranges[f][-1])
    # ylim(0.0, 4.0)
    title(titles[i])
    xticks([ranges[f][1], ranges[f][2]])
    ylim(0, 2)
    if i == 0:
        ylabel("Rate\n(norm.)")

    ################
    # PEARSON Correlation
    ################
    gs_corr3 = gridspec.GridSpecFromSubplotSpec(2,1, gs2[i,2], hspace=0.9, wspace=0.2)
    
    subplot(gs_corr3[0,0])
    simpleaxis(gca())

    corr3 = []
    base3 = []    
    for j, keys in enumerate(orders):
        st, gr, sd, k = keys

        corr2 = corr[st][gr][sd]
        corr2 = corr2[corr2['n']>4][k]
        idx = corr2.index.values
        corr2 = corr2.values.astype(float)
        corr3.append(corr2)
        base3.append(baseline[st][gr][sd][k].loc[idx].values.astype(float))

        plot(np.random.randn(len(corr2))*0.05+np.ones(len(corr2))*(j+1), corr2, '.', markersize=1)
    
    
    xlim(0.5, 2.5)
    gca().spines['bottom'].set_bounds(1, 2)
    ymin = corr3[0].min()
    ylim(ymin-0.1, 1.1)
    gca().spines['left'].set_bounds(ymin-0.1, 1.1)

    ylabel("Pearson r")
    xticks([1, 2], ['Chrimson', 'Tdtomato'], fontsize=fontsize-1)

    # if i == 1: 
    #     yticks([])
    #     gca().spines["left"].set_visible(False)

    ############
    subplot(gs_corr3[1,0])
    simpleaxis(gca())
    xlim(0.5, 2.5)
    ylim(-0.5, 1)
    gca().spines['bottom'].set_bounds(1, 2)
    xlabel("minus baseline")
    # if i == 1: gca().spines["left"].set_visible(False)
    plot([1,2],[0,0], linestyle='--', color=COLOR, linewidth=0.1)
    # tmp = (corr[['decimated', 'down']]-baseline[['decimated', 'down']]).values.astype("float")
    tmp = [c-b for c, b in zip(corr3, base3)]
    vp = violinplot(tmp, showmeans=False, 
        showextrema=False, vert=True, side='high'
        )    
    # sys.exit()
    # vp = violinplot(corr3, showmeans=False,
    #     showextrema=False, vert=True, side='high'
    #     )
    for k, p in enumerate(vp['bodies']): p.set_color(cycle[k])

    m = [c.mean() for c in corr3]
    plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

    xticks([1,2],['',''])
    ylabel(r"Mean$\Delta$")
    # if i == 1: 
    #     yticks([])
    #     gca().spines["left"].set_visible(False)






outergs.update(top=0.95, bottom=0.05, right=0.98, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/fig2.pdf",
    dpi=200,
    facecolor="white",
)
# show()

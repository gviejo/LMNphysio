# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-05-31 18:08:14
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
    fig_height = fig_width * golden_mean * 0.75  # height in inches
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

rcParams["font.family"] = 'Liberation Sans'
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

# COmputing tests
map_significance = {
    1:"n.s.",
    2:"*",
    3:"**",
    4:"***"
}


###############################################################################################
# LOADING DATA
###############################################################################################
dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"




###############################################################################################################
# PLOT
###############################################################################################################

markers = ["d", "o", "v"]

fig = figure(figsize=figsize(1))

outergs = GridSpec(2, 1, hspace = 0.5)


names = {'adn':"ADN", 'lmn':"LMN"}
epochs = {'wak':'Wake', 'sws':'Sleep'}



gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[0, 0], width_ratios=[0.2, 0.8], wspace=0.3
)


#####################################
# HISTO
#####################################

subplot(gs_top[0,0])

axvline(0, linewidth=0.5, linestyle='--', color=COLOR)
text(0.5, 1.0, 'Midline', ha='center', bbox=dict(facecolor=None, edgecolor=None, alpha=0), transform=gca().transAxes)

box_width, box_height = 0.75, 0.4
y_positions = [1, 2, 3]
x_positions = [-0.6, 0.6]

box_colors = [colors[st] for st in ['lmn', 'adn', 'psb']]
ax = gca()
ax.set_xlim(-2, 2)
ax.set_ylim(0.5, 3.5)
# ax.set_aspect('equal')
# ax.axis('off')

# Draw boxes
for i, x in enumerate(x_positions):
    for j, y in enumerate(y_positions):
        ax.text(x, y, ['LMN', 'ADN', 'PSB'][j], 
            ha='center', va='center',
            bbox=dict(facecolor=box_colors[j], edgecolor=box_colors[j], boxstyle="round,pad=0.3")
            )

# Draw reversed vertical arrows using FancyArrow (Box 3 → 2 → 1)
for i in range(2):
    start_y = y_positions[i]
    for j in range(2):
        annotate("", (x_positions[j], y_positions[i+1]), (x_positions[j], y_positions[i]+box_height/2),
            arrowprops=dict(arrowstyle="simple", facecolor='gray', fc="gray", lw=0)
            )

annotate("", (x_positions[1],y_positions[1]-box_height/2), (x_positions[0], y_positions[0]+box_height/2),
    xycoords='data', textcoords='data',
    arrowprops=dict(arrowstyle="simple", facecolor='gray', fc="gray", lw=0))

annotate("", (x_positions[0],y_positions[1]-box_height/2), (x_positions[1], y_positions[0]+box_height/2),
    xycoords='data', textcoords='data',
    arrowprops=dict(arrowstyle="simple", facecolor='gray', fc="gray", lw=0))


# Right-angle arrow from Box 1 → Box 3 using FancyArrowPatch
signs = [1, -1]
for i, x_position in enumerate(x_positions):
    x = x_position - (box_width/2 - 0.1)*signs[i]
    # annotate("", 
    #     # (x+signs[i]*0.1, y_positions[0]),
    #     # (x+signs[i]*0.1, y_positions[-1]),
    #     (0.5, 3),
    #     (0.5, 1),
    #     arrowprops=dict(
    #         arrowstyle="->", 
    #         facecolor='gray', 
    #         fc="gray", 
    #         lw=0,
    #         connectionstyle=f"bar,fraction=0.3")
    #         # connectionstyle=f"bar,fraction={0.2*signs[i]}")
    #     )

    arrow = FancyArrowPatch(
        posA=(x+signs[i]*0.1, y_positions[-1]),     # Right of top box
        posB=(x+signs[i]*0.1, y_positions[0]),     # Right of bottom box
        connectionstyle=f"bar,fraction={0.2*signs[i]}",       # Top down with bend
        arrowstyle="->,head_length=1,head_width=1",
        color="gray",
        linewidth=1,
    )
    ax.add_patch(arrow)



# subplot(gs_histo[0,0])
# noaxis(gca())
# img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/LMN-PSB-opto.png")
# imshow(img, aspect="equal")
# xticks([])
# yticks([])


# subplot(gs_histo[1,0])
# noaxis(gca())
# img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/A8066_S7_3_2xMerged.png")
# imshow(img, aspect="equal")
# xticks([])
# yticks([])


# subplot(gs_histo[2,0])
# noaxis(gca())
# img = mpimg.imread(os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/A8066_S7_3_4xMerged.png")
# imshow(img, aspect="equal")
# xticks([])
# yticks([])

#####################################
# Examples ADN OPTO
#####################################
st = 'adn'

gs_raster_ex = gridspec.GridSpecFromSubplotSpec(1, 2, gs_top[0,1], hspace=0.5, wspace=0.3)


exs = {
    'wak': {
        'ipsi': ("B3700/B3704/B3704-240609A", nap.IntervalSet(5130, 5232)),
        "bilateral": ("B2800/B2810/B2810-240925B", nap.IntervalSet(8269, 8379))
        },
    'sws': {
        "ipsi": ("B3700/B3704/B3704-240608A", nap.IntervalSet(4110.6, 4117.5)),
        "bilateral": ("B2800/B2809/B2809-240904B", nap.IntervalSet(3897.29, 3905.7))
        }
    }


for i, epoch in enumerate(['wak', 'sws']):

    gs_ex = gridspec.GridSpecFromSubplotSpec(2, 2, gs_raster_ex[0,i], hspace=0.5, wspace=0.9, height_ratios=[1, 0.8])

    for j, k in enumerate(['ipsi', 'bilateral']):

        s, ex = exs[epoch][k]

        path = os.path.join(data_directory, "OPTO", s)

        spikes, position, wake_ep, opto_ep, sws_ep = load_opto_data(path, st)

                
        subplot(gs_ex[0,j])
        simpleaxis(gca())    
                
        plot(spikes.to_tsd("order").restrict(ex), '|', color=colors[st], markersize=0.7, mew=0.25)


        s, e = opto_ep.intersect(ex).values[0]
        rect = patches.Rectangle((s, len(spikes)+1), width=e-s, height=1,
            linewidth=0, facecolor=opto_color)
        gca().add_patch(rect)
        [axvline(t, color=COLOR, linewidth=0.1, alpha=0.5) for t in [s,e]]

        ylim(0, len(spikes)+2)
        xlim(ex.start[0], ex.end[0])
        xticks([])
        yticks([0, len(spikes)-1], [1, len(spikes)])
        gca().spines['left'].set_bounds(0, len(spikes)-1)
        gca().spines['bottom'].set_bounds(s, e)
        title(epochs[epoch])
        ylabel(names[st])
    

        #
        exex = nap.IntervalSet(ex.start[0] - 10, ex.end[0] + 10)
        p = spikes.count(0.01, exex).smooth(0.04, size_factor=20)
        d=np.array([p.loc[i] for i in spikes.index[np.argsort(spikes.order)]]).T
        p = nap.TsdFrame(t=p.t, d=d, time_support=p.time_support)
        p = np.sqrt(p / p.max(0))
        # p = 100*(p / p.max(0))

        subplot(gs_ex[1,j])
        simpleaxis(gca())
        tmp = p.restrict(ex)
        d = gaussian_filter(tmp.values, 2)
        tmp2 = nap.TsdFrame(t=tmp.index.values, d=d)

        im = pcolormesh(tmp2.index.values, 
                np.linspace(0, 2*np.pi, tmp2.shape[1]),
                tmp2.values.T, cmap='GnBu', antialiased=True)

        x = np.linspace(0, 2*np.pi, tmp2.shape[1])
        yticks([0, 2*np.pi], [1, len(spikes)])
        
        gca().spines['bottom'].set_bounds(s, e)
        xticks([s, e], ['', ''])
        if i == 0: xlabel("10 s", labelpad=-1)
        if i == 1: xlabel("1 s", labelpad=-1)

        if epoch == "wake":
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



# ##########################################
# # ADN OPTO
# ##########################################

gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[1,0]
    )




# # gs_corr = gridspec.GridSpecFromSubplotSpec(
# #     2, 2, subplot_spec=gs_bottom[0, 2], hspace=1.0#, height_ratios=[0.5, 0.2, 0.2] 
# # )

# allorders = {"OPTO_SLEEP": [('adn', 'opto', 'ipsi', 'opto'), 
#                         ('adn', 'opto', 'ipsi', 'sws'),],
#             "OPTO_WAKE": [('adn', 'opto', 'ipsi', 'opto'), 
#                         ('adn', 'opto', 'ipsi', 'pre'),]}


# ranges = {
#     "OPTO_SLEEP":(-0.9,0,1,1.9),
#     "OPTO_WAKE":(-9,0,10,19)
#     }
ranges = (-0.9,0,1,1.9)

data = cPickle.load(open(os.path.expanduser("~/Dropbox/LMNphysio/data/OPTO_SLEEP.pickle"), 'rb'))
allr = data['allr']
corr = data['corr']
change_fr = data['change_fr']
allfr = data['allfr']
baseline = data['baseline']


titles = ['Ipsilateral', 'Bilateral']

for i, f in enumerate(['ipsi', 'bilateral']):

    gs_corr2 = gridspec.GridSpecFromSubplotSpec(2,2, gs_bottom[0,i], hspace=1)#, width_ratios=[0.2, 0.1])

    orders = ('adn', 'opto', f, 'opto')

    ####################
    # FIRING rate change
    ####################    
    subplot(gs_corr2[0,0])
    simpleaxis(gca())

    s, e = ranges[1], ranges[2]
    rect = patches.Rectangle((s, 1.8), width=e-s, height=0.2,
        linewidth=0, facecolor=opto_color)
    gca().add_patch(rect)
    [axvline(t, color=COLOR, linewidth=0.1, alpha=0.5) for t in [s,e]]

    keys = orders
    tmp = allfr[keys[0]][keys[1]][keys[2]]
    tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1.5, mode='constant'))
    tmp = tmp.loc[ranges[0]:ranges[-1]]
    m = tmp.mean(1)
    s = tmp.std(1)
    
    # plot(tmp, color = 'grey', alpha=0.2)
    plot(tmp.mean(1), color = COLOR, linewidth=0.75)
    fill_between(m.index.values, m.values-s.values, m.values+s.values, color=COLOR, alpha=0.2, ec=None)
    
    xlabel("Opto. time (s)", labelpad=1)
    xlim(ranges[0], ranges[-1])
    # ylim(0.0, 4.0)
    title(titles[i])
    xticks([ranges[1], ranges[2]])
    ylim(0, 2)    
    ylabel("Rate\n(norm.)")

    subplot(gs_corr2[1,0])
    simpleaxis(gca())

    # ch_fr = change_fr[keys[0]][keys[1]][keys[2]]['opto']

    chfr = change_fr[keys[0]][keys[1]][keys[2]]
    delta = (chfr['opto'] - chfr['pre'])/chfr['pre']    

    num_bins = 30
    bins = np.linspace(-1, 1, num_bins + 1)
    
    hist(delta.values, bins=bins, histtype="stepfilled", facecolor="#404040", edgecolor="#404040")
    axvline(0, color=COLOR, linestyle='--', linewidth=0.5)

    ylabel("%", labelpad=1)
    xlabel("% mod. rate", labelpad=1)
    ylim(0, 30)

    zw, p = scipy.stats.ttest_1samp(np.array(delta.values).astype(float), popmean=0)
    signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    text(0.6, 0.8, s=map_significance[signi], va="center", ha="left", transform=gca().transAxes)


    colors2 = [opto_color, "#FF7F50"]


    ###############################
    # CHRIMSON CHRIMSON COMPARAISON
    ###############################
    orders2 =   [('adn', 'opto', f, 'opto'),
                ('adn', 'opto', f, 'decimated')]

    #
    ax1 = subplot(gs_corr2[0,1])
    simpleaxis(gca())
    title("Matching FR")
    gca().spines['bottom'].set_bounds(1, 2)
    # gca().spines['left'].set_visible(False)
    # yticks([])

    corr3 = []
    base3 = []    
    for j, keys in enumerate(orders2):
        st, gr, sd, k = keys

        corr2 = corr[st][gr][sd]
        
        tokeep = corr2['n']>4

        corr2 = corr2[tokeep][k]

        idx = corr2.index.values        

        corr3.append(corr2.values.astype(float))

        base3.append(baseline[st][gr][sd][k].loc[idx].values.astype(float))
        


    corr3 = pd.DataFrame(data=np.array(corr3).T, columns=['opto', 'decimated'])
    base3 = pd.DataFrame(data=np.array(base3).T, columns=['opto', 'decimated'])

    for s in corr3.index:
        plot([1, 2], corr3.loc[s,['opto', 'decimated']], '-', color=COLOR, linewidth=0.1)
    plot(np.ones(len(corr3)),   corr3['opto'], 'o', color=opto_color, markersize=1)
    plot(np.ones(len(corr3))*2, corr3['decimated'], 'o', color=COLOR, markersize=1)

    
    ymin = corr3['opto'].min()
    ylim(ymin-0.1, 1.1)
    xlim(0.5, 3)
    # gca().spines['left'].set_bounds(ymin-0.1, 1.1)

    # ylabel("Pearson r")
    xticks([1, 2], ['Chrimson', 'Control'], fontsize=fontsize-1)

    #
    ax2 = subplot(gs_corr2[1,1])
    simpleaxis(gca())
    # gca().spines['left'].set_visible(False)
    # yticks([])        
    ylim(-0.5, 1)
    xlim(0.5, 3)
    gca().spines['bottom'].set_bounds(1, 2)
    xlabel("minus baseline", labelpad=1)
    plot([1,2.2],[0,0], linestyle='--', color=COLOR, linewidth=0.2)
    plot([2.2], [0], 'o', color=COLOR, markersize=0.5)


    tmp = corr3 - base3

    print(tmp)

    vp = violinplot(tmp, showmeans=False, 
        showextrema=False, vert=True, side='high'
        )
    colors4 = [opto_color, COLOR]
    # for k, p in enumerate(vp['bodies']):
    #     p.set_color(colors4[k])
    #     p.set_alpha(1)

    m = tmp.mean(0).values
    plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

    xticks([1,2],['',''])
    # ylabel(r"Mean$\Delta$")

    # COmputing tests
    for i in range(2):
        zw, p = scipy.stats.mannwhitneyu(corr3.iloc[:,i], base3.iloc[:,i], alternative='greater')
        signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
        text(i+0.9, m[i]-0.07, s=map_significance[signi], va="center", ha="right")
    
    xl, xr = 2.5, 2.6
    plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
    plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
    plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
    zw, p = scipy.stats.mannwhitneyu(tmp['opto'].dropna(), tmp['decimated'].dropna())
    signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    text(xr+0.1, np.mean(m)-0.07, s=map_significance[signi], va="center", ha="left")







outergs.update(top=0.92, bottom=0.09, right=0.98, left=0.1)


savefig(
    os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/fig3.pdf",
    dpi=200,
    facecolor="white",
)
# show()

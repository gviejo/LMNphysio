# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-07-28 11:30:36
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
from scipy.optimize import curve_fit

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
    fig_height = fig_width * golden_mean * 0.95  # height in inches
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

# rcParams["font.family"] = 'Liberation Sans'
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

outergs = GridSpec(2, 1, hspace = 0.6, height_ratios=[0.5,0.5])


names = {'adn':"ADN", 'lmn':"LMN"}
epochs = {'wak':'Wake', 'sws':'Sleep'}

Epochs = ['Wake', 'Sleep']



gs_top = gridspec.GridSpecFromSubplotSpec(1,3, outergs[0,0], 
    hspace = 0.45, wspace = 0.5, width_ratios=[0.3, 0.4, 0.2])


#####################################
# Example
#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(3,3, gs_top[0,0], 
    hspace = 0.2, wspace = 0.2, width_ratios=[0.2, 0.5, 0.5])


dropbox_path = os.path.expanduser("~") + "/Dropbox/LMNphysio/data"

filepath = os.path.join(dropbox_path, "DATA_FIG_LMN_ADN_A5011-201014A.pickle")
data = cPickle.load(open(filepath, 'rb'))


tcurves = data['tcurves']
angle = data['angle']
peaks = data['peaks']
spikes = data['spikes']
lmn = data['lmn']
adn = data['adn']
wake_ep = data['wake_ep']
sws_ep = data['sws_ep']


exs = {'wak':nap.IntervalSet(start = 7587976595.668784, end = 7604189853.273991, time_units='us'),
        'sws':nap.IntervalSet(start = 15038.3265, end = 15039.4262, time_units = 's')}
neurons={'adn':adn,'lmn':lmn}

tokeep = np.sort(np.hstack((adn,lmn)))
decoded, P = nap.decode_1d(tcurves[tokeep], spikes[tokeep], exs['sws'], 0.01)

peak = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

n_adn = peak[adn].sort_values().index.values[-4]
n_lmn = peak[lmn].sort_values().index.values[-8]

ex_neurons = [n_adn, n_lmn]


for j, e in enumerate(['wak', 'sws']):

    subplot(gs1[0,j+1])
    simpleaxis(gca())
    gca().spines['bottom'].set_visible(False)

    for i, st in enumerate(['adn', 'lmn']):
        if e == 'wak':
            angle2 = angle
        if e == 'sws':
            angle2 = decoded

        spk = spikes[ex_neurons[i]]
        isi = nap.Tsd(t = spk.index.values[0:-1], d=np.diff(spk.index.values))
        idx = angle2.as_series().index.get_indexer(isi.index, method="nearest")
        isi_angle = pd.Series(index = angle2.index.values, data = np.nan)
        isi_angle.loc[angle2.index.values[idx]] = isi.values
        isi_angle = isi_angle.ffill()

        isi_angle = nap.Tsd(isi_angle)
        isi_angle = isi_angle.restrict(exs[e])        

        # isi_angle = isi_angle.value_from(isi, exs[e])
        semilogy(isi_angle, '-', color = colors[st], linewidth = 0.5, markersize = 0.5, alpha=1)
    
    xlim(exs[e].loc[0,'start'], exs[e].loc[0,'end'])
    ylim(0.001, 10)

    xticks([])
    if j == 0: 
        yticks([0.001, 0.1, 10], [0.001, 0.1, 10])
    else:
        yticks([0.001, 0.1, 10], ["", "", ""])

    title(Epochs[j])
    if j == 0: 
        ylabel('ISI (s)')#, rotation =0, y=0.4, labelpad = 15)

for i, st in enumerate(['adn', 'lmn']):

    subplot(gs1[i+1, 0])
    simpleaxis(gca())
    tmp = tcurves[ex_neurons[i]]
    tmp = tmp / tmp.max()
    plot(tmp.values, tmp.index.values, linewidth = 1, color = colors[st])

    # gca().invert_xaxis()
    # gca().yaxis.tick_right()
    gca().spines['left'].set_bounds(0, 2*np.pi)
    # gca().spines['top'].set_visible(False)
    yticks([0, 2*np.pi], [0, 360])
    ylabel(names[st], rotation=0)
    # xticks([tmp.values.max()], [str(int(tmp.values.max()))])
    
    if i == 1:
        xticks([0, 1], [0, 100])
        xlabel("Rate (%)")
    else:
        xticks([])


    for j, e in enumerate(['wak', 'sws']):
        subplot(gs1[i+1,j+1]) 
        simpleaxis(gca())
        ylim(0, 2*np.pi)
        xticks([])
        if e == 'wak':
            tmp = angle.restrict(exs[e])
            tmp = tmp.as_series().rolling(window=40, win_type='gaussian', center=True, min_periods=1).mean(std=2.0)
            plot(tmp, linewidth = 1, color = COLOR, label = 'H.D.')
        if e == 'sws':
            tmp2 = decoded
            tmp2 = smoothAngle(tmp2, 2)
            tmp2 = tmp2.restrict(exs[e])
            iset=np.abs(np.gradient(tmp2)).threshold(1.0, method='below').time_support
            for a, b in iset.values:
                plot(tmp2.get(a, b), linestyle='dashdot', linewidth=1, color=COLOR)
            plot(tmp2.get(a, b), linestyle='dashdot', linewidth=1, color=COLOR, label="Decoded H.D.")


        n = ex_neurons[i]
        spk = spikes[n].restrict(exs[e]).index.values   
        #clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
        plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = colors[st], markersize = 3, markeredgewidth = 0.4)
        yticks([])
        xlim(exs[e].loc[0,'start'], exs[e].loc[0,'end'])
        if i == 1 and j == 0:
            # xlabel(str(int(exs[e].tot_length('s')))+' s', horizontalalignment='center')#, x=1.0)
            s, e = exs[e].start[0], exs[e].end[0]
            gca().spines['bottom'].set_bounds(e-5,e)
            xticks([e-5, e], ["", ""])
            text(e-2.5, -2.2, s="5 sec.", va="center", ha="center")

        elif i == 1 and j == 1:
            # xlabel(str(int(exs[e].tot_length('s')))+' s', horizontalalignment='center')#, x=1.0)
            s, e = exs[e].start[0], exs[e].end[0]
            gca().spines['bottom'].set_bounds(e-0.4,e)
            xticks([e-0.4, e], ["", ""])
            text(e-0.2, -2.2, s="0.4 sec.", va="center", ha="center")
        else:
            gca().spines['bottom'].set_visible(False)

        # if j == 0:
        yticks([0, 2*np.pi], [])
            # ylabel(names[i], rotation=0)
        if i == 1:
            if j == 0:
                legend(handlelength = 1.1, frameon=False, bbox_to_anchor=(0.4, -0.9, 0.5, 0.5))
            else:
                legend(handlelength = 1.5, frameon=False, bbox_to_anchor=(0.9, -0.9, 0.5, 0.5))








########################################################################################
# ISI HD MAPS
########################################################################################
data = cPickle.load(open(os.path.join(dropbox_path, 'ALL_LOG_ISI.pickle'), 'rb'))
logisi = data['logisi']
frs = data['frs']

pisi = {'adn':cPickle.load(open(os.path.join(dropbox_path, 'PISI_ADN.pickle'), 'rb')),
        'lmn':cPickle.load(open(os.path.join(dropbox_path, 'PISI_LMN.pickle'), 'rb'))}



mkrstype = ['-', '--']

gs2 = gridspec.GridSpecFromSubplotSpec(3,4, gs_top[0,1], 
    wspace = 0.4, hspace = 0.2, 
    height_ratios=[0.1,0.2,0.2], width_ratios=[0.5, 0.5, 0.02, 0.3])

for j, e in enumerate(['wak', 'sws']):
    subplot(gs2[0,j])
    simpleaxis(gca())
    for i, st in enumerate(['adn', 'lmn']):
        tc = pisi[st]['tc_'+e]
        tc = tc/tc.max(0)
        m = tc.mean(1)
        s = tc.std(1)
        plot(m, mkrstype[j], label = names[st], color = colors[st], linewidth = 1)
        fill_between(m.index.values,  m-s, m+s, color = colors[st], alpha = 0.1)
        
        ylim(0, 1.01)
        xticks([-np.pi, 0, np.pi], ['', '', ''])
        xlim(-np.pi, np.pi)
        
    if j==0:
        ylabel(r"Rate (%)")
        yticks([0, 1], [0, 100])
    else:
        yticks([0, 1], ['', ''])
    title(epochs[e])
    # if j==1:
    #     legend(handlelength = 0.6, frameon=False, bbox_to_anchor=(1.2, 0.55, 0.5, 0.5))

# Precompute map

pisihd = {}
minmax = []
for i, st in enumerate(['adn', 'lmn']):
    pisihd[st] = {}
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
minmax = np.array(minmax)

for i, st in enumerate(['adn', 'lmn']):
    
    for j, e in enumerate(['wak', 'sws']):
        subplot(gs2[i+1,j])

        im = imshow(pisihd[st][e], cmap = 'bwr', 
            aspect= 'auto', origin='lower')

        yticks(xt, ['',''])

        if i == 0:            
            xticks([0, tmp3.shape[1]//2, tmp3.shape[1]-1], ['', '', ''])
        if i == 1:
            xticks([0, tmp3.shape[1]//2, tmp3.shape[1]-1], ['-180', '0', '180'])
            xlabel('Centered HD')

        if j == 1:            
            yticks(xt, ['', ''])
            # ylabel(names[st], rotation=0, labelpad=8)
            # text(names[st], 1, 0.5)
        if j == 0:
            yticks(xt, ['0.01', '1'])
            ylabel('ISI (s)')
            text(-1.0, 0.5, s=names[st], va="center", ha="left", transform=gca().transAxes)


# Colorbar
axip = gca().inset_axes([1.2, 2.3, 0.1, 0.7])
# noaxis(axip)
cbar = colorbar(im, cax=axip)
axip.set_title("%", y=0.8)




for i, st in enumerate(['adn', 'lmn']):    
    subplot(gs2[i+1,3])
    simpleaxis(gca())    
    linestyle = ['-', '--']    

    for j, e in enumerate(['wak', 'sws']):
        cut = -40
        bins = pisi[st]['bins'][0:cut]        
        tmp = pisi[st][e].mean(0)[0:cut]
        y = tmp.mean(1)
        y = (y/y.sum())*100
        semilogy(y, bins[0:-1], linestyle=linestyle[j], linewidth=0.8, color=colors[st],
            label=epochs[e]
            )

    yticks([0.01, 1], ['0.01', '1'])
    xlim(0, 1.8)
    if i == 0:
        xticks([0, 1], ['',''])
        legend(handlelength = 1.1, frameon=False, bbox_to_anchor=(1.2, 1.4, 0.5, 0.5))
    if i == 1:
        xticks([0, 1])
        xlabel("%")



        


#########################################
# Gaussian Mixture fit
#########################################

data = cPickle.load(open(os.path.join(dropbox_path, 'All_ISI.pickle'), 'rb'))

pr2 = data['pr2']




gs3 = gridspec.GridSpecFromSubplotSpec(2,2, gs_top[0,2], 
    wspace = 0.6, hspace = 0.9, height_ratios=[0.5, 0.5])



bins = np.geomspace(0.001, 10.0, 50)

# for i, st in enumerate(['adn']):
st = 'adn'
subplot(gs3[0,:])
simpleaxis(gca())
    # for j, ep in enumerate([sws_ep]):

td = spikes[ex_neurons[0]].time_diff(epochs=sws_ep)

counts, edges = np.histogram(td.values, bins)
bin_widths = np.diff(edges)
pmf = (counts/counts.sum())*100

fill_between(bins[0:-1], pmf, step='post', alpha=0.5, color='grey', edgecolor='None')
plot(bins[0:-1], pmf, drawstyle='steps-post', color='k', linewidth=0)  # outline

# Plotting the fit
# x_plot = np.linspace(np.log(bins.min()), np.log(bins.max()), 100).reshape(-1, 1)
x_plot = np.log(bins).reshape(-1, 1)
from sklearn.mixture import GaussianMixture
gmms = []
for n_components in [1, 2]:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(np.log(td.values)[:,None])
    gmms.append(gmm)

colors2 = ['#1b9e77', '#d95f02']

for k, gmm in enumerate(gmms):
    logprob = gmm.score_samples(x_plot)
    pdf = np.exp(logprob)
    pdf = (pdf/pdf.sum())*100
    plot(bins, pdf, linewidth=1, color=colors2[k], alpha=0.5)

# legend()
    if k == 0:
        annotate(f"$Ll_1={np.round(np.sum(logprob),1)}$", 
            xy=(bins[25], pdf[25]), 
            xytext = (bins[34], pdf[25]-0.8),
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.1,
                color=COLOR,
                shrinkB=0,
                shrinkA=0.1
                ),
            ha="left",
            va="center",
            fontsize=fontsize-2
            )
    if k == 1:
        annotate(f"$Ll_2={np.round(np.sum(logprob),1)}$", 
            xy=(bins[16], pdf[16]), 
            xytext = (bins[31], pdf[16]-0.4),
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.1,
                color=COLOR,
                shrinkB=0,
                shrinkA=0.1
                ),
            ha="left",
            va="center",
            fontsize=fontsize-2
            )


gca().spines['bottom'].set_bounds(bins[0], bins[-1])
xscale("log")
title("Ex. " + names[st] + "\n Gaussian Mixture", y=0.8)
xticks([0.001, 0.1, 10], ['0.001', '0.1', '10'])
xlabel("ISI (s)")
ylabel("Prop. (%)")

#################
# PR2
#################


for j, e in enumerate(['wak', 'sws']):

    subplot(gs3[1,j])

    simpleaxis(gca())    

    title(epochs[e], y=0.9)

    tmp = [pr2[st][e] for st in ['adn', 'lmn']]
    
    vp = violinplot(tmp, [1,2], showmeans=False, 
        showextrema=False, vert=True)#, side='high')

    for k, p in enumerate(vp['bodies']):
        p.set_color(colors[['adn','lmn'][k]])
        p.set_alpha(1)

    m = [tmp[i].mean(0) for i in range(2)]
    plot([1, 2], m, 'o', markersize=0.5, color=COLOR)

    xticks([1, 2], ['ADN', 'LMN'], rotation=45)
    ylim(-0.01, 0.3)

    # # COmputing tests    
    # for i in range(2):
    #     zw, p = scipy.stats.wilcoxon(tmp[i])
    #     signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    #     text(i+0.9, m[i]-0.01, s=map_significance[signi], va="center", ha="right")

    xl, xr = 2.5, 2.6
    plot([xl, xr], [m[0], m[0]], linewidth=0.2, color=COLOR)
    plot([xr, xr], [m[0], m[1]], linewidth=0.2, color=COLOR)
    plot([xl, xr], [m[1], m[1]], linewidth=0.2, color=COLOR)
    zw, p = scipy.stats.mannwhitneyu(tmp[0].dropna(), tmp[1].dropna())
    signi = np.digitize(p, [1, 0.05, 0.01, 0.001, 0.0])
    text(xr+0.1, np.mean(m)-0.01, s=map_significance[signi], va="center", ha="left")

    if j == 1:
        yticks([0.0, 0.3], ['', ''])
    else:
        yticks([0.0, 0.3])
        ylabel(r"Bimodality ($pR^{2}$)")
            
    gca().spines['bottom'].set_bounds(1, 2)

    print("mannwhitneyu", e, zw, p)

# ##########################################
# BOTTOM
# ##########################################
# MODEL

gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outergs[1,0], width_ratios = [0.4, 0.8], wspace=0.1
    )

def ellipse_points(t, center, width, height, angle=0):
    x0, y0 = center
    a, b = width / 2, height / 2
    theta = np.deg2rad(angle)
    x = a * np.cos(t)
    y = b * np.sin(t)
    # Apply rotation
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    # Translate to center
    x_final = x0 + x_rot
    y_final = y0 + y_rot
    return x_final, y_final

# Diagram
ax = subplot(gs_bottom[0,0])
title("Model")
noaxis(ax)
from matplotlib.patches import Ellipse
# First oval LMN
height = 0.04
width = 0.1
y_lmn = 0.1

oval1 = Ellipse((0.5, y_lmn), width=width, height=height, angle=0,
                edgecolor=colors['lmn'], facecolor='none', linewidth=1)
ax.add_patch(oval1)
x1, y1 = ellipse_points([-np.pi/2], (0.5, y_lmn), width, height)
ax.text(0.5-width/2-0.03, y_lmn, "LMN", 
    ha='right', va='center', fontsize=fontsize-1,
    bbox=dict(facecolor=colors['lmn'], edgecolor='none', boxstyle='round,pad=0.2')
    )

# Second oval ADN
width = 0.3
y_adn = 0.5
oval2 = Ellipse((0.5, y_adn), width=width, height=height, angle=0,
                edgecolor=colors['adn'], facecolor='none', linewidth=1, zorder=2)
ax.add_patch(oval2)
a = -np.pi/2
offset = (2*np.pi)/36
tmp = np.linspace(a-offset*3, a+offset*3, 5)
x2, y2 = ellipse_points(tmp, (0.5, y_adn), width, height)
ax.text(0.5-width/2-0.03, y_adn, "ADN", 
    ha='right', va='center', fontsize=fontsize-1,
    bbox=dict(facecolor=colors['adn'], edgecolor='none', boxstyle='round,pad=0.2')
    )


alphas = [0.5, 0.75, 1, 0.75, 0.5]
lws = [0.5, 0.75, 1, 0.75, 0.5]
for i in range(len(x2)):
    # plot([x1[0], x2[i]], [y1[0], y2[i]], linewidth=lws[i], color='grey', alpha=alphas[i])
    arrow = FancyArrowPatch(
        [x1[0], y1[0]], [x2[i], y2[i]],
        arrowstyle="->",
        color=COLOR,
        linewidth=lws[i],
        # alpha=alphas[i],
        mutation_scale=5,
        zorder=2
        )
    ax.add_patch(arrow)

plot(x1, y1, 'o', color=colors['lmn'], markersize=2)
plot(x2, y2, 'o', color=colors['adn'], markersize=2)


# Inset axes non linearity
axi = ax.inset_axes([0.2, y_lmn+(y_adn-y_lmn)/2-0.01, 0.2, 0.1])
simpleaxis(axi)
x = np.linspace(-20, 20, 100)
axi.plot(x, (1/(1+np.exp(-x))), linewidth=1, color=COLOR)
axi.set_xticks([])
axi.set_yticks([])
axi.set_xlabel(r"$I_{LMN}$")
axi.set_ylabel(r"$x_{ADN}$")

# Set limits and aspect
ax.set_xlim(-0.1, 1)
ax.set_ylim(0, 0.86)
# ax.set_aspect('equal')

# ADN -> PSB
y_psb = 0.75

arrow = FancyArrowPatch(
    [0.5, y_adn-0.01], [0.5, y_psb],
    arrowstyle="-",
    color=COLOR,
    linewidth=lws[i],
    # alpha=alphas[i],
    mutation_scale=5,
    zorder=2
    )
ax.add_patch(arrow)

# PSB Feedback
ax.text(0.5, y_psb, "PSB Feedback", 
    ha='center', va='center', fontsize=fontsize-1,
    bbox=dict(facecolor=colors['psb'], edgecolor='none', boxstyle='round,pad=0.2')
    )

# Define start and end points
start = (0.6, y_psb)
end = (0.55, y_lmn)
arrow = FancyArrowPatch(
    start, end,
    connectionstyle="arc3,rad=-0.4",  # curvature (positive: left curve, negative: right)
    arrowstyle="->",
    color=COLOR,
    linewidth=0.5,
    mutation_scale=5
)
ax.add_patch(arrow)

# Inh
x_inh = 0.75
ax.plot(x_inh, y_adn, 'o', color=COLOR, markersize=3)
ax.text(x_inh+0.085, y_adn, "Inh.", 
    ha='center', va='center', fontsize=fontsize-1,
    bbox=dict(facecolor="lightgrey", edgecolor='none', boxstyle='round,pad=0.2')
    )
arrow = FancyArrowPatch(
    (x_inh, y_adn), (x_inh-0.1, y_adn),
    connectionstyle="arc3,rad=-0.6",  # curvature (positive: left curve, negative: right)
    arrowstyle="-[",
    color=COLOR,
    linewidth=0.5,
    mutation_scale=1,
    zorder=2
)
ax.add_patch(arrow)

arrow = FancyArrowPatch(
    (x_inh-0.1, y_adn), (x_inh, y_adn),
    connectionstyle="arc3,rad=-0.6",  # curvature (positive: left curve, negative: right)
    arrowstyle="->",
    color=COLOR,
    linewidth=0.5,
    mutation_scale=5,
    zorder=2
)
ax.add_patch(arrow)


box_height = 0.1
box_widths = [0.61, 1.0, 0.7]
x_lefts = [0.0, -0.08, 0.02]
facecolors = ["None", "white", "None"]
# Draw boxes
for i, y in enumerate([y_lmn, y_adn, 0.75]):
    outerrect = patches.FancyBboxPatch((x_lefts[i], y - box_height/2),
                                   box_widths[i], box_height,
                                   boxstyle="round,pad=0.01",
                                   edgecolor=COLOR,
                                   facecolor=facecolors[i], linewidth=0.5, linestyle='--')
    ax.add_patch(outerrect)
    ax.text(x_lefts[i]+0.12, y, 
        ['Mammilary\nBody', 'Anterior\nThalamus', 'Cortex'][i], ha='center', va='center', fontsize=4)



######################################
# Activity
######################################

filepath = os.path.join(os.path.expanduser("~") + "/Dropbox/LMNphysio/model/model.pickle")
data = cPickle.load(open(filepath, 'rb'))
popcoh = data['popcoh']

import matplotlib.colors as mcolors
cmaps = {}
for name, hex_color in colors.items():
    # Custom colormap from white to the color
    cmaps[name] = mcolors.LinearSegmentedColormap.from_list(name, ['white', hex_color])


gs_bottom2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_bottom[0,1], wspace=0.4, width_ratios=[0.6, 0.01, 0.4]
    )


gs_activity = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_bottom2[0,0], wspace=0.5
    )


durations = [2000, 1000, 100]
offset = 100

titles = ["'Wake'", "'Sleep'", "'Opto.'\n No PSB feedback"]

for i, st in enumerate(['adn', 'lmn']):
    for j, e in enumerate(['wak', 'sws', 'opto']):
        subplot(gs_activity[i,j])
        simpleaxis(gca())

        # start = int(sl.start + (sl.stop - sl.start)/2)

        # tmp = data[st][start:start+durations[j]].T
        tmp = data[st][e][offset:offset+durations[j]]
        tmp = tmp/tmp.max()

        # im=imshow(tmp.T, origin='lower', aspect='auto', cmap=cmaps[st])
        im=imshow(tmp.T, origin='lower', aspect='auto', cmap='bwr', vmin=0, vmax=1)

        if j == 0:
            ylabel(st.upper())
            yticks([0, tmp.shape[1]])            
        else:
            yticks([0, tmp.shape[1]], ["", ""])

        if i == 1:            
            xticks([0, durations[j]], [0, durations[j]])
        else:
            xticks([0, durations[j]], ["", ""])
            title(titles[j])


        if i == 1 and j == 1:
            xlabel("Simulation steps")

        if j == 2:
            # Colorbar
            axip = gca().inset_axes([1.3, 0.0, 0.1, 0.6])
            # noaxis(axip)
            cbar = colorbar(im, cax=axip)
            axip.set_title("Rate", y=1.0)
            axip.set_yticks([0, 1])


######################################
# Population Coherence
######################################
gs_popcoh = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_bottom2[0,2], wspace=0.04
    )

# Real data

corrs = pd.DataFrame(index=['adn-sws', 'lmn-sws', 'adn-opto', 'lmn-opto'], columns=['mean', 'std'])

data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_LMN.pickle'), 'rb'))
allrlmn = data['allr']
r_lmn = data['pearsonr']
data = cPickle.load(open(os.path.join(dropbox_path, 'All_correlation_ADN.pickle'), 'rb'))
allradn = data['allr']
r_adn = data['pearsonr']
allr_sess = {'adn':r_adn, 'lmn':r_lmn}
for k, r in allr_sess.items():
    corrs.loc[k+"-sws", 'mean'] = allr_sess[k]['sws'].mean()
    corrs.loc[k+"-sws", 'std'] = allr_sess[k]['sws'].std()

data = cPickle.load(open(os.path.expanduser(f"~/Dropbox/LMNphysio/data/OPTO_SLEEP.pickle"), 'rb'))

corrs.loc["lmn-opto","mean"] = data['corr']['lmn']['opto']['ipsi']['opto'].mean()
corrs.loc["lmn-opto","std"] = data['corr']['lmn']['opto']['ipsi']['opto'].std()

corrs.loc["adn-opto","mean"] = data['corr']['adn']['opto']['bilateral']['opto'].mean()
corrs.loc["adn-opto","std"] = data['corr']['adn']['opto']['bilateral']['opto'].std()


xx = [[0, 1], [0, 1]]


for i, st in enumerate(['adn', 'lmn']):

    subplot(gs_popcoh[i,0])
    simpleaxis(gca())
    gca().spines['bottom'].set_bounds(0, 1)
    gca().spines['left'].set_bounds(-0.25, 1)

    m = corrs.loc[[f"{st}-sws", f"{st}-opto"], "mean"]
    s = corrs.loc[[f"{st}-sws", f"{st}-opto"], "std"]
    # print(st, m)
    plot(xx[i], m, '-', color=colors[st])
    errorbar(xx[i], m, yerr=s, fmt='o', 
            elinewidth=0.5,              # Error bar line width
            capsize=1,                 # Length of the cap
            capthick=0.5,              # Cap line thickness
            alpha=1,                 # Transparency
            markersize=2,               # Size of the marker
            color=colors[st])

    ylim(-0.25, 1.05)
    xlim(-0.5, 1.5)

    plot([0, 1], [0, 0], linestyle="--", color=COLOR, linewidth=0.1)

    if i == 0:
        ylabel("Pop. coherence (r)", y=0, labelpad=5)
        title("Observed")
        xticks([])

    if i == 1:
        xticks([0, 1], ["Sleep", "Opto."])
    else:
        xticks([0, 1], ["", ""])


# Model
from scipy.stats import pearsonr

for i, st in enumerate(['adn', 'lmn']):  
    ax = subplot(gs_popcoh[i,1]) 
    simpleaxis(ax)   
    ax.spines["left"].set_visible(False)
    gca().spines['bottom'].set_bounds(0, 1)
    gca().spines['left'].set_bounds(-0.25, 1)
    yticks([])
    r = [pearsonr(popcoh[st]['wak'], popcoh[st][k])[0] for j, k in enumerate(['sws','opto'])]

    plot(xx[i], r, 'o-', color=colors[st], markersize=2)

    ylim(-0.25, 1.05)
    xlim(-0.5, 1.5)

    plot([0, 1], [0, 0], linestyle="--", color=COLOR, linewidth=0.1)

    if i == 0:
        title("Model")
        xticks([])

    if i == 1:
        xticks([0, 1], ["\'Sleep\'", "\'Opto.\'"])
    else:
        xticks([0, 1], ["", ""])



outergs.update(top=0.94, bottom=0.09, right=0.98, left=0.06)


# savefig(
#     os.path.expanduser("~") + "/Dropbox/LMNphysio/paper2024/fig4.pdf",
#     dpi=200,
#     facecolor="white",
# )
# show()


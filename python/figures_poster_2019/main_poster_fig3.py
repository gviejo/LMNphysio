#!/usr/bin/env python


import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
import sys
sys.path.append("../")
from functions import *
from wrappers import *
from pylab import *
import _pickle as cPickle
import neuroseries as nts
import os
import hsluv
# from mtspec import mtspec, wigner_ville_spectrum
from scipy.stats import linregress


tcurves = pd.read_hdf("../../figures/figures_poster_2019/alltcurves.h5")

hd_sessions = ['A1407-190416', 'A1407-190417', 'A1407-190422']

hd1 = [s for s in tcurves.columns if s.split("_")[0] in hd_sessions[0]]

allahv = pd.read_hdf('../../figures/figures_poster_2019/allahv.h5')

ump = pd.read_hdf('../../figures/figures_poster_2019/ump.h5')

###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.4          # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.set_xticks([])
	ax.set_yticks([])
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)



import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	# "text.usetex": True,                # use LaTeX to write all text
	# "font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 7,
	"ytick.labelsize": 7,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.8,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}  
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(1.0))

outergs = GridSpec(2,1, figure = fig, height_ratios = [0.5,0.5], hspace = 0.35)


####################################################################
# A EXEMPLES
####################################################################
gs_top = gridspec.GridSpecFromSubplotSpec(3,5, subplot_spec = outergs[0,0], wspace = 0.1, hspace = 0.2, width_ratios = [0.1, 0.1, 0.01, 0.1, 0.1])

# EXEMPLES HD
for i in range(3):
	subplot(gs_top[i,0], projection = 'polar')
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	yticks([])
	n = hd1[0]
	tmp = tcurves[n].values
	tmp /= tmp.max()
	fill_between(tcurves[n].index.values, np.zeros_like(tmp), tmp , color = 'red', alpha = 0.5, linewidth =1, zorder=2)

	subplot(gs_top[i,1])
	plot(allahv[n])

# EXEMPLES NON-HD
for i in range(3):
	subplot(gs_top[i,3], projection = 'polar')
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	yticks([])
	n = hd1[0]
	tmp = tcurves[n].values
	tmp /= tmp.max()
	fill_between(tcurves[n].index.values, np.zeros_like(tmp), tmp , color = 'red', alpha = 0.5, linewidth =1, zorder=2)

	subplot(gs_top[i,4])
	plot(allahv[n])



####################################################################
# B UMAP
####################################################################
gs_bottom = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[1,0], width_ratios = [0.5, 0.2])
subplot(gs_bottom[0,0])

noaxis(gca())

labels = ump['label'].values

colors = np.array(['red', 'blue', 'orange', 'green', 'purple'])

colors = np.array(['#00aeef', '#6ec1e4', '#1f5673', '#212d40', '#114b5f'])


scatter(ump[0], ump[1], c = colors[labels])

xlim(-13, 15)
ylim(-5, 11)
ax = gca()

w1, w2 = (.2, .3)
lw = 1
tmp = pd.read_hdf('../../figures/figures_poster_2019/allahv_normalized.h5')

##############################################################################
axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(.6, -.04, w1, w2),
                   bbox_transform=ax.transAxes, loc=3)
l = 3
plot(tmp.iloc[:,labels == l], color = colors[0], alpha = 0.7, linewidth =lw)
title('0')
simpleaxis(gca())

##############################################################################
axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(.1, -.05, w1, w2),
                   bbox_transform=ax.transAxes, loc=3)
l = 0
plot(tmp.iloc[:,labels == l], color = colors[1], alpha = 0.7, linewidth =lw)
title('1')
simpleaxis(gca())

##############################################################################
axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(.01, .5, w1, w2),
                   bbox_transform=ax.transAxes, loc=3)
l = 4
plot(tmp.iloc[:,labels == l], color = colors[2], alpha = 0.7, linewidth =lw)
simpleaxis(gca())
title('2')

##############################################################################
axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(.4, .85, w1, w2),
                   bbox_transform=ax.transAxes, loc=3)
l = 2
plot(tmp.iloc[:,labels == l], color = colors[3], alpha = 0.7, linewidth =lw)
title('3')
simpleaxis(gca())

##############################################################################
axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(.8, .5, w1, w2),
                   bbox_transform=ax.transAxes, loc=3)
l = 1
plot(tmp.iloc[:,labels == l], color = colors[4], alpha = 0.7, linewidth =lw)
simpleaxis(gca())


outergs.update(top= 0.95, bottom = 0.05, right = 0.95, left = 0.02)
savefig("../../figures/figures_poster_2019/fig_poster_3.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_poster_2019/fig_poster_3.pdf &")

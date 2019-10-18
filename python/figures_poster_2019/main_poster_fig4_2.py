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
from mpl_toolkits.mplot3d import Axes3D

# LMN
data = cPickle.load(open('../../figures/figures_poster_2019/fig_4_ring_lmn.pickle', 'rb'))

ump_lmn = data['ump']
ang_lmn = data['wakangle']

data = cPickle.load(open('../../figures/figures_poster_2019/fig_4_ring_adn.pickle', 'rb'))

ump_adn = data['ump']
ang_adn = data['wakangle']


###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1          # height in inches
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

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


fig = figure(figsize = figsize(1.0))

ax = fig.add_subplot(111)
noaxis(ax)

H = ang_adn.values/(2*np.pi)
HSV = np.vstack((H*360, np.ones_like(H)*85, np.ones_like(H)*45)).T
RGB = np.array([hsluv.hsluv_to_rgb(HSV[i]) for i in range(len(HSV))])

ax.scatter(ump_adn[~np.isnan(H),0]*0.4+6, ump_adn[~np.isnan(H),1]*0.5+6, c = RGB[~np.isnan(H)], marker = '.', alpha = 0.9, linewidth = 0, s= 40, zorder = 3)


H = ang_lmn.values/(2*np.pi)
HSV = np.vstack((H*360, np.ones_like(H)*90, np.ones_like(H)*45)).T
RGB = np.array([hsluv.hsluv_to_rgb(HSV[i]) for i in range(len(HSV))])

ax.scatter(ump_lmn[~np.isnan(H),0]*0.4, ump_lmn[~np.isnan(H),1]*1, c = RGB[~np.isnan(H)], marker = '.', alpha = 0.9, linewidth = 0, s= 50, zorder = 1)

# ax.view_init(azim=30, elev = 35)

ax.text(7.5, -14,  "LMn", fontsize = 20, fontweight='bold')

ax.text(0, 6, "ADn", fontsize = 20, fontweight='bold')

ax.text(5.5, -2,  "?", fontsize = 20, fontweight='bold')

x, y = (np.mean(ump_lmn[:,0]*0.4), np.mean(ump_lmn[:,1]*1))
# ax.arrow(x, y, x+1, y+1)
x2, y2 = (np.mean(ump_adn[:,0]*0.4+6), np.mean(ump_adn[:,1]*0.5+6))

ax.annotate("", xy=(x2, y2), xytext=(x, y), arrowprops=dict(arrowstyle="-|>",lw=2), zorder = 2, alpha = 0.6)


# a = Arrow3D([8, 1], [-10, 1], 
#             [0.01, 1], mutation_scale=20, 
#             lw=4, arrowstyle="-|>", color="grey", alpha = 1, linewidth = 0)
# ax.add_artist(a)
# a.zorder = 3

# colorbar
from matplotlib.colorbar import ColorbarBase
colors = [hsluv.hsluv_to_hex([i,85,45]) for i in np.arange(0, 361)]
cmap= matplotlib.colors.ListedColormap(colors)
cax = inset_axes(ax, "15%", "3%",
                   bbox_to_anchor=(0.8, 0.35, 1, 1),
                   bbox_transform=ax.transAxes, 
                   loc = 'lower left')
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=matplotlib.colors.Normalize(vmin=0,vmax=360),
                                orientation='horizontal')
cb1.set_ticks([0,360])
cb1.set_ticklabels(['0', '360'])
cax.set_title("Wake", pad = 3)

# tight_layout()

savefig("../../figures/figures_poster_2019/fig_poster_4.pdf", dpi = 200, facecolor = 'white')
os.system("evince ../../figures/figures_poster_2019/fig_poster_4.pdf &")

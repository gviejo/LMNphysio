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
from sklearn.cluster import KMeans



data = cPickle.load(open('../../figures/figures_poster_2019/all_tcurves_AD_LMN.pickle', 'rb'))

# # hd_sessions = ['A1407-190416', 'A1407-190417', 'A1407-190422']

# hd1 = [s for s in tcurves.columns if s.split("_")[0] in hd_sessions[0]]

# allahv = pd.read_hdf('../../figures/figures_poster_2019/allahv.h5')

ump = pd.read_hdf('../../figures/figures_poster_2019/ump.h5')

lmn_hdc = data['lmn_hdc']
lmn_ahv = data['lmn_ahv'].loc[-2.1:2.1]
adn_hdc = data['adn_hdc']
adn_ahv = data['adn_ahv'].loc[-2.1:2.1]

nucleus_groups = ump.groupby('nucleus').groups

lmn_hd = ump.loc[nucleus_groups['lmn']].groupby('hd').groups[1]

lmn_no = ump.loc[nucleus_groups['lmn']].groupby('hd').groups[0]
lmn_lb = ump.loc[lmn_no].groupby('labels').groups


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

outergs = GridSpec(2,1, figure = fig, height_ratios = [0.5,0.4], hspace = 0.35)


####################################################################
# A EXEMPLES
####################################################################
gs_top = gridspec.GridSpecFromSubplotSpec(3,5, subplot_spec = outergs[0,0], wspace = 0.05, hspace = 0.2, width_ratios = [0.1, 0.1, 0.01, 0.1, 0.1])

ex1 = [2, 8, 15]
#8
# EXEMPLES HD
for i in range(3):
	subplot(gs_top[i,0], projection = 'polar')
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	yticks([])
	if i == 0:
		n = nucleus_groups['ad'][ex1[i]]
		fill_between(adn_hdc[n].index.values, np.zeros_like(adn_hdc[n].values), adn_hdc[n].values , color = 'red', alpha = 0.7, linewidth =0, zorder=2)
		ylabel("ADn", rotation = 0, labelpad = 20, fontsize = 12)
	else:
		n = lmn_hd[ex1[i]]
		fill_between(lmn_hdc[n].index.values, np.zeros_like(lmn_hdc[n].values), lmn_hdc[n].values , color = 'grey', alpha = 0.7, linewidth =0, zorder=2)
		plot(lmn_hdc[n], color = 'dodgerblue')
		if i == 1:
			ylabel("LMn", rotation = 0, labelpad = 20, fontsize = 12)

	subplot(gs_top[i,1])
	simpleaxis(gca())
	if i == 0:		
		plot(adn_ahv[n], color = 'red', alpha =0.7)		
	else:
		plot(lmn_ahv[n], color = 'grey', alpha = 0.7)
	if i == 1:
		yticks([10, 15])
	if i == 2:
		xlabel("Angular Head Velocity (deg/s)")
	if i == 1:
		ylabel("Firing rate (Hz)")

	xticks([-(np.pi*100)/180, 0, (np.pi*100)/180], [-100, 0, 100])


gr2 = [1, 4, 2]
ex2 = [1, 0, 44]
# EXEMPLES NON-HD
for i in range(3):
	subplot(gs_top[i,3], projection = 'polar')
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	yticks([])
	n = lmn_lb[gr2[i]][ex2[i]]
	fill_between(lmn_hdc[n].index.values, np.zeros_like(lmn_hdc[n].values), lmn_hdc[n].values , color = 'grey', alpha = 0.7, linewidth =1, zorder=2)

	subplot(gs_top[i,4])
	simpleaxis(gca())
	plot(lmn_ahv[n], color = 'grey', alpha = 0.7)

	if i == 1:
		yticks([15, 20])
	if i == 2:
		xlabel("Angular Head Velocity (deg/s)")
	if i == 1:
		ylabel("Firing rate (Hz)")

	xticks([-(np.pi*100)/180, 0, (np.pi*100)/180], [-100, 0, 100])

####################################################################
# B UMAP
####################################################################

# normalized tuning curves to plot the average
# data = cPickle.load(open('../../figures/figures_poster_2019/all_tcurves_AD_LMN.pickle', 'rb'))

gs_bottom = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outergs[1,0])#, width_ratios = [0.5, 0.2])
subplot(gs_bottom[0,0])

noaxis(gca())

labels = ump['labels'].values

colors = np.array(['#000000', '#969696', '#7b7d7b', '#bfbfbf'])


labels = KMeans(n_clusters = 4, random_state = 0).fit(ump.loc[nucleus_groups['lmn'], [0,1]]).labels_

scatter(ump.loc[nucleus_groups['ad'], [0]], ump.loc[nucleus_groups['ad'], [1]], color = 'red', alpha = 0.7, linewidth = 0)
for n in np.unique(labels):
	scatter(ump.loc[nucleus_groups['lmn'][labels == n], [0]], ump.loc[nucleus_groups['lmn'][labels == n], [1]], color = colors[n], alpha = 0.9, linewidth = 0)

scatter(ump.loc[lmn_hd, 0], ump.loc[lmn_hd, 1], color = 'dodgerblue', alpha = 1, s = 2)

xlim(-15, 15)
ylim(-5, 12)
ax = gca()

w1, w2 = (.18, .32)
lw = 1

# lmn_ahv = pd.read_hdf('../../figures/figures_poster_2019/allahv_normalized.h5')

# adn_ahv = pd.read_hdf('../../figures/figures_poster_2019/allahv_normalized.h5')

data = cPickle.load(open('../../figures/figures_poster_2019/all_tcurves_AD_LMN.pickle_normalized', 'rb'))

lmn_ahv = data['lmn_ahv'].loc[-2.1:2.1]
adn_ahv = data['adn_ahv'].loc[-2.1:2.1]


##############################################################################
# AD
##############################################################################
axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(0.05, -0.01, w1, w2),
                   bbox_transform=ax.transAxes, loc=3)

x = adn_ahv.loc[-2.1:2.1].index.values
m = adn_ahv.loc[-2.1:2.1].mean(1)
s = adn_ahv.loc[-2.1:2.1].sem(1)
plot(x, m, color = 'red', alpha = 1)
fill_between(x, m-s, m+s, color = 'red', alpha= 0.5)
title('ADn')
xlabel("Angular Head Velocity\n(deg/s)", fontsize = 6, labelpad = -0.6)
ylabel("Norm. rate (a.u.)", fontsize = 6)

xticks([-(np.pi*100)/180, 0, (np.pi*100)/180], [-100, 0, 100], fontsize = 6)	
yticks(fontsize = 6)

##############################################################################
# LMN
##############################################################################

pos = [	[0.4, 0.8],
		[0.7, 0.6],
		[0.65, 0.00],
		[0.1, 0.7]]

for n in np.unique(labels):
	##############################################################################
	axins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(pos[n][0], pos[n][1], w1, w2), bbox_transform=ax.transAxes, loc=3)
	
	idx = nucleus_groups['lmn'][labels == n]
	x = lmn_ahv.loc[-2.1:2.1,idx].index.values
	m = lmn_ahv.loc[-2.1:2.1,idx].mean(1)
	s = lmn_ahv.loc[-2.1:2.1,idx].sem(1)
	plot(x, m, color = colors[n])
	fill_between(x, m-s, m+s, color = colors[n], alpha = 0.7)
	# title(n)	
	# yticks([1])

	if n == 0:
		xlabel("Angular Head Velocity (deg/s)", fontsize = 6, labelpad = -0.6)
	else :
		xlabel("Angular Head Velocity\n(deg/s)", fontsize = 6, labelpad = -0.6)
	ylabel("Norm. rate (a.u.)", fontsize = 6)

	xticks([-(np.pi*100)/180, 0, (np.pi*100)/180], [-100, 0, 100], fontsize = 6)	
	yticks(fontsize = 6)

outergs.update(top= 0.95, bottom = 0.05, right = 0.98, left = 0.02)
savefig("../../figures/figures_poster_2019/fig_poster_3.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_poster_2019/fig_poster_3.pdf &")

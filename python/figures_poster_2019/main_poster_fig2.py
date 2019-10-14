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



def smoothAngle(tsd, sd):
	tmp 			= pd.Series(index = tsd.index.values, data = np.unwrap(tsd.values))	
	tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=sd)
	newtsd			= nts.Tsd(tmp2%(2*np.pi))
	return newtsd

# data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
data_directory 		= '../../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

session = 'A1407-190416'

data = cPickle.load(open('../../figures/figures_poster_2019/fig_2_crosscorr.pickle', 'rb'))

tcurves		 		= data['tcurves']
pairs 				= data['pairs']
sess_groups	 		= data['sess_groups']
frates		 		= data['frates']
cc_wak		 		= data['cc_wak']
cc_rem		 		= data['cc_rem']
cc_sws		 		= data['cc_sws']
peaks 				= data['peaks']

idx = [n for n in pairs.index.values if session in n[0] and session in n[1]]
pairs = pairs.loc[idx]
groups = pairs.groupby(np.digitize(pairs, [0, np.pi/3, 2*np.pi/3, np.pi])-1).groups

# data = cPickle.load(open('../../figures/figures_poster_2019/fig_1_decoding.pickle', 'rb'))

# tcurves = data['tcurves']

# peaks = data['peaks']

# neurons = tcurves.columns.values

pair_index = [1, 2, 2]


###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.2          # height in inches
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

outergs = GridSpec(2,1, figure = fig, height_ratios = [0.75,0.25], hspace = 0.35)


####################################################################
# A EXEMPLES
####################################################################
gs_top = gridspec.GridSpecFromSubplotSpec(4,5, subplot_spec = outergs[0,0], wspace = 0.3, hspace = 0.5, width_ratios = [0.1, 0.1, 0.1, 0.01, 0.1])

for i in range(3):
	n1, n2 = groups[i][pair_index[i]]
	subplot(gs_top[0,i], projection = 'polar')
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	yticks([])
	for n in [n1, n2]:
		clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
		tmp = tcurves[n].values
		tmp /= tmp.max()
		fill_between(tcurves[n].index.values, np.zeros_like(tmp), tmp , color = clr, alpha = 0.5, linewidth =1, zorder=2)	

for i, epoch, cc in zip(range(3), ['WAKE', 'REM', 'NREM'], [cc_wak, cc_rem, cc_sws]):
	for j in range(3):
		nn = groups[j][pair_index[j]]
		subplot(gs_top[i+1,j])
		simpleaxis(gca())
		# plot(cc[nn])
		tmp = cc[nn]
		fill_between(tmp.index.values, np.zeros_like(tmp.values), tmp.values, color = 'grey', alpha = 0.6, linewidth = 0)
		xticks(fontsize = 6)
		yticks(fontsize = 6)
		if j == 0:
			ylabel(epoch)
		if i == 2:
			xlabel("Time lag (ms)", labelpad = -0.8, fontsize = 7)



####################################################################
# B MEAN FIRING RATE
####################################################################

subplot(gs_top[0:2,-1])
simpleaxis(gca())
loglog(frates['wake'].values, frates['rem'].values, 'o', color = 'black', markersize = 3)
slope, intercept, r, p, stderr = linregress(np.log(frates['wake'].values), np.log(frates['rem'].values))
x = np.arange(np.log(frates['wake'].values.min()), np.log(frates['wake'].values.max()+10))
y = slope*x + intercept
loglog(np.exp(x), np.exp(y), color = 'red', alpha = 0.6, linewidth = 1)
xlabel("Wake Firing rate (Hz)", fontsize =6, labelpad = -0.5)
ylabel("REM Firing rate (Hz)", fontsize = 6)
xticks(fontsize=6)
yticks(fontsize=6)

subplot(gs_top[2:4,-1])
simpleaxis(gca())
loglog(frates['wake'].values, frates['sws'].values, 'o', color = 'black', markersize = 3)
slope, intercept, r, p, stderr = linregress(np.log(frates['wake'].values), np.log(frates['sws'].values))
x = np.arange(np.log(frates['wake'].values.min()), np.log(frates['wake'].values.max()+10))
y = slope*x + intercept
loglog(np.exp(x), np.exp(y), color = 'red', alpha = 0.6, linewidth = 1)
xlabel("Wake Firing rate (Hz)", fontsize =6, labelpad = -0.5)
ylabel("NREM Firing rate (Hz)", fontsize = 6)
xticks(fontsize=6)
yticks(fontsize=6)


####################################################################
# C AVERAGE CROSS CORR
####################################################################
gs_bottom = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec = outergs[1,0], width_ratios = [0.1, 0.5, 0.5, 0.5], wspace = 0.4)#, height_ratios = [0.2, 0.8], hspace = 0)

# angular differences
subplot(gs_bottom[:,0])
simpleaxis(gca())
plot(pairs.values, np.arange(len(pairs))[::-1])
xticks([0, np.pi], ['0', r'$\pi$'], fontsize = 6)
yticks([0, len(pairs)-1], [len(pairs), 1], fontsize = 6)
xlabel("Angular\ndifference\n(rad)", fontsize = 7, labelpad = -0.5)
ylabel("Pairs")
ylim(0, len(pairs)-1)

for i, epoch, cc in zip(range(3), ['WAKE', 'REM', 'NREM'], [cc_wak, cc_rem, cc_sws]):
	subplot(gs_bottom[:,i+1])
	simpleaxis(gca())
	imshow(cc[pairs.index].T, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
	times = cc.index.values
	xticks([0, np.where(times==0)[0], len(times)], [times[0], 0, times[-1]], fontsize = 6)	
	yticks([0, len(pairs)-1], [1, len(pairs)], fontsize = 6)
	title(epoch, pad = -1)
	xlabel("Time lag (ms)", labelpad = -0.8, fontsize = 7)




outergs.update(top= 0.95, bottom = 0.1, right = 0.95, left = 0.07)
savefig("../../figures/figures_poster_2019/fig_poster_2.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_poster_2019/fig_poster_2.pdf &")

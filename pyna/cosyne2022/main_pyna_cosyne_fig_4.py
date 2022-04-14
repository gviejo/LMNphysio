# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 17:49:50
# @Last Modified by:   gviejo
# @Last Modified time: 2022-03-16 00:48:16
import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.pyplot import *
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import sys
sys.path.append('../')
from functions import *
from scipy.ndimage import gaussian_filter

def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*0.5      # height in inches
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



###############################################################################################################
# PLOT
###############################################################################################################
# mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	# "text.usetex": True,                # use LaTeX to write all text
	# "font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 21,               # LaTeX default is 10pt font.
	"font.size": 21,
	"legend.fontsize": 21,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 15,
	"ytick.labelsize": 15,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 5,
	"axes.linewidth"        : 1.5,
	"ytick.major.size"      : 4,
	"xtick.major.size"      : 4
	}  
mpl.rcParams.update(pdf_with_latex)

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(2.0))

outergs = gridspec.GridSpec(1, 3, figure=fig, wspace = 0.3)#, hspace = 0.1)

names = ['ADN', 'LMN']
clrs = ['sandybrown', 'olive']

Epochs = ['Wake', 'non-REM sleep']
Epochs2 = ['Wake', 'non-REM']

path2 = '/home/guillaume/Dropbox/CosyneData'

#################################################################################################
########################################################################################
# IO
########################################################################################
subplot(outergs[0])
simpleaxis(gca())

IO_fr = pd.read_hdf(path2+'/IO_fr.hdf')

plot(IO_fr['lmn'], 'o-', color = clrs[1], label = 'Integrator', markersize = 3)
plot(IO_fr['adn'], 'o-', color = clrs[0], label = 'Activator', markersize = 3)


legend(handlelength = 0.8, frameon=False, bbox_to_anchor=(0.5, -0.1, 0.5, 0.5))

xlabel("Input (Hz)")
ylabel("Ouput (Hz)")

########################################################################################
# CC
########################################################################################
titles = ['Integrator', 'Activator']
clrs = ['#E66100', '#5D3A9B']

data = cPickle.load(open(os.path.join(path2, 'MODEL_CC.pickle'), 'rb'))

for i, st in enumerate(['int-int', 'adn-adn']):
	subplot(outergs[i+1])
	simpleaxis(gca())
	for j in range(2):
		m = data[st][j]['m']
		s = data[st][j]['s']
		plot(m.index.values*1000, m.values, color = clrs[j], linewidth = 3)
		fill_between(m.index.values*1000, m-s, m+s, alpha = 0.2, color = clrs[j])
	gca().spines['left'].set_position('center')
	gca().text(0.6, 0.96, 'z', horizontalalignment='center', verticalalignment='center', transform=gca().transAxes)
	xlabel('cross. corr. (ms)')	
	xticks([-100,0,100])
	#yticks([])
	locator_params(axis='y', nbins=3)


	title(titles[i], pad = 13)

# Inset axes
#######################################
ax1 = fig.add_axes([0.57, 0.75, 0.2, 0.2], polar=True)


ax2 = fig.add_axes([0.57, 0.53, 0.2, 0.2], polar=True)

iax = [ax1, ax2]

path2 = '/home/guillaume/Dropbox/CosyneData'

exn = [[50,-2],[0,-1],[1,-2]]

acc = cPickle.load(open(os.path.join(path2, 'All_crosscor_ADN_LMN.pickle'), 'rb'))
allpairs = acc['pairs']
cc_sws = acc['cc_sws']
tcurves = acc['tcurves']

accadn = cPickle.load(open(os.path.join(path2, 'All_crosscor_ADN_adrien.pickle'), 'rb'))

allpairs = pd.concat([allpairs, accadn['pairs']])
cc_sws = pd.concat((cc_sws, accadn['cc_sws']), 1)
tcurves = pd.concat((tcurves, accadn['tcurves']), 1)


subpairs = allpairs[allpairs['struct']=='adn-adn']
group = subpairs.sort_values(by='ang diff').index.values

angdiff = allpairs.loc[group,'ang diff'].values.astype(np.float32)
group2 = group[angdiff<np.deg2rad(40)]
group3 = group[angdiff>np.deg2rad(140)]
pos2 = np.where(angdiff<np.deg2rad(40))[0]
pos3 = np.where(angdiff>np.deg2rad(140))[0]
clrs = ['red', 'green']
clrs = ['#E66100', '#5D3A9B']
idx = [group2[exn[0][0]], group3[exn[0][1]]]
i = 0
# Tuning curves
for j in range(2):
	cax1 = iax[j]
	tmp = tcurves[list(idx[j])]
	tmp = tmp.dropna()
	if i == 0: 
		tmp = tmp.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)			
	tmp = tmp/tmp.max()
	cax1.plot(tmp.iloc[:,0], linewidth = 2, color = clrs[j])
	cax1.plot(tmp.iloc[:,1], linewidth = 2, color = clrs[j])		
	cax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
	cax1.set_xticklabels([])
	cax1.set_yticks([])	



outergs.update(top= 0.87, bottom = 0.15, right = 0.95, left = 0.1)

#show()
savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2022 poster/figures/fig4.pdf", dpi = 100, facecolor = 'white')



# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 17:02:39
# @Last Modified by:   gviejo
# @Last Modified time: 2022-03-15 12:43:56
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
import pynapple as nap
from pylab import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.0         # height in inches
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

fig = figure(figsize = figsize(2))

# outergs = GridSpec(1,1, figure = fig)
gs1 = GridSpec(5, 3, figure=fig, height_ratios = [0.2, -0.1, 0.3, 0.05, 0.6], wspace = 0.2, hspace = 0.3)

#################################################################################################################################
# gs1 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec = outergs[0,0], height_ratios = [0.3, 0.6], wspace = 0.1, hspace = 0.3)

names = ['ADN/ADN', 'LMN/ADN', 'LMN/LMN']
ks = ['adn-adn', 'adn-lmn', 'lmn-lmn']
clrs = ['lightgray', 'gray', 'darkgray']

#xlabels = ['ADN/ADN (ms)', 'LMN/ADN (ms)', 'LMN/LMN (ms)']

path2 = '/home/guillaume/Dropbox/CosyneData'


acc = cPickle.load(open(os.path.join(path2, 'All_crosscor_ADN_LMN.pickle'), 'rb'))
allpairs = acc['pairs']
cc_sws = acc['cc_sws']
tcurves = acc['tcurves']

accadn = cPickle.load(open(os.path.join(path2, 'All_crosscor_ADN_adrien.pickle'), 'rb'))

allpairs = pd.concat([allpairs, accadn['pairs']])
cc_sws = pd.concat((cc_sws, accadn['cc_sws']), 1)
tcurves = pd.concat((tcurves, accadn['tcurves']), 1)
letters = ['A', 'B', 'C']
exn = [[2,-2],[0,-1],[1,-2]]

for i, n in enumerate(names):

	subpairs = allpairs[allpairs['struct']==ks[i]]
	group = subpairs.sort_values(by='ang diff').index.values

	angdiff = allpairs.loc[group,'ang diff'].values.astype(np.float32)
	group2 = group[angdiff<np.deg2rad(40)]
	group3 = group[angdiff>np.deg2rad(140)]
	pos2 = np.where(angdiff<np.deg2rad(40))[0]
	pos3 = np.where(angdiff>np.deg2rad(140))[0]
	clrs = ['red', 'green']
	clrs = ['#E66100', '#5D3A9B']
	idx = [group2[exn[i][0]], group3[exn[i][1]]]

	# Tuning curves
	gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, gs1[0,i])
	for j in range(2):
		cax1 = subplot(gs2[0,j], projection='polar')
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
	
	## Example Cross-corr
	axD = subplot(gs1[2,i])
	simpleaxis(gca())
	gca().spines['left'].set_position('center')		
	inset_pos = [-0.1, 0.1]
	for j in range(2):		
		# Cross correlogram SWS			
		cc = cc_sws[idx[j]].loc[-200:200]
		cc = (cc - cc.mean())/cc.std()
		cc = cc.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)			
		plot(cc, linewidth = 2, color = clrs[j])		

	# title(names[i], pad = 70)
	gca().text(0.5, 1.8, names[i], horizontalalignment='center', verticalalignment='center', transform=gca().transAxes)
	#gca().text(-0.0, 1.15, letters[i], horizontalalignment='center', verticalalignment='center', transform=gca().transAxes, fontsize = 21, weight="bold")


	#xlabel(xlabels[i])
	#if i == 2: xlabel("cc. (ms)")
	gca().set_ylabel("z",  y = 0.8, rotation = 0, labelpad = 15)
	yticks([])
	xticks([-100,0,100])


	## MEAN CC
	subplot(gs1[4,i])
	simpleaxis(gca())
	for j,gr in enumerate([group2, group3]):
		cc = cc_sws[gr]		
		cc = cc - cc.mean(0)
		cc = cc / cc.std(0)
		cc = cc.loc[-200:200]
		m  = cc.mean(1)
		s = cc.std(1)
		plot(cc.mean(1), color = clrs[j], linewidth = 3)
		fill_between(cc.index.values, m - s, m+s, color = clrs[j], alpha = 0.1)
	gca().spines['left'].set_position('center')
	xlabel('cross. corr. (ms)')	
	xticks([-100,0,100])
	#yticks([])
	locator_params(axis='y', nbins=3)
	# ylabel('z',  y = 0.9, rotation = 0, labelpad = 1)
	gca().text(0.45, 1.0, 'z', horizontalalignment='center', verticalalignment='center', transform=gca().transAxes)
	#######################################
	cax1 = inset_axes(gca(), "20%", "30%",					
	                   bbox_to_anchor=(0.05, 0.7, 1, 1),
	                   bbox_transform=gca().transAxes, 
	                   loc = 'lower left',	                   
	                   )
			
	plot(np.arange(len(group)), allpairs.loc[group, 'ang diff'].values, color = 'grey')
	for j,gr,ps in zip(range(2),[group2, group3],[pos2,pos3]):
		plot(ps, allpairs.loc[gr, 'ang diff'].values, '-', color = clrs[j], linewidth = 3)
	if i == 1: xlabel("Pairs")
	if i == 1: ylabel("Ang. \n diff.", rotation = 0, y = 0.1)
	xticks([])
	yticks([])
	simpleaxis(gca())
	yticks([0, np.pi], [0, 180])
	axhspan(0, np.deg2rad(40), color =  clrs[0], alpha = 0.3)
	axhspan(np.deg2rad(140), np.deg2rad(180), color = clrs[1], alpha = 0.3)
	#######################################
	





# outergs.update(top= 0.93, bottom = 0.1, right = 0.98, left = 0.02)
gs1.update(top= 0.95, bottom = 0.1, right = 0.98, left = 0.02)

savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2022 poster/figures/fig2.pdf", dpi = 100, facecolor = 'white')


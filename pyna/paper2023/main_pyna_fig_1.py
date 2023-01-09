# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-12-30 15:29:06
import numpy as np
import pandas as pd
import pynapple as nap

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.font_manager as font_manager
#matplotlib.style.use('seaborn-paper')
import matplotlib.image as mpimg

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv

import os
import sys
sys.path.append('../')
from functions import *
sys.path.append('../../python')
import neuroseries as nts


def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
	#fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_width = 6
	fig_height = fig_width*golden_mean*1         # height in inches
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

font_dir = ['/home/guillaume/Dropbox/CosyneData/figures_poster_2022']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

fontsize = 8

COLOR = (0.25, 0.25, 0.25)

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = fontsize
rcParams['text.color'] = COLOR
rcParams['axes.labelcolor'] = COLOR
rcParams['axes.labelsize'] = fontsize
rcParams['axes.labelpad'] = 3
#rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = fontsize
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize
rcParams['figure.titlesize'] = fontsize
rcParams['xtick.major.size'] = 1.3
rcParams['ytick.major.size'] = 1.3
rcParams['xtick.major.width'] = 0.4
rcParams['ytick.major.width'] = 0.4
rcParams['axes.linewidth'] = 0.2
rcParams['axes.edgecolor'] = COLOR
rcParams['axes.axisbelow'] = True
rcParams['xtick.color'] = COLOR
rcParams['ytick.color'] = COLOR

############################################################################################### 
# GENERAL infos
###############################################################################################
name = 'A5011-201014A'
path = '/home/guillaume/Dropbox/CosyneData/A5011-201014A'

path2 = '/home/guillaume/Dropbox/CosyneData'


############################################################################################### 
# LOADING DATA
###############################################################################################
data = nap.load_session(path, 'neurosuite')

spikes = data.spikes#.getby_threshold('freq', 1.0)
angle = data.position['ry']
position = data.position
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')
rem_ep = data.read_neuroscope_intervals('rem')

# Only taking the first wake ep
wake_ep = wake_ep.loc[[0]]

adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 60, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 1)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI)
spikes = spikes.getby_threshold('SI', 0.1, op = '>')
tuning_curves = tuning_curves[spikes.keys()]

tokeep = list(spikes.keys())

adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values

tcurves = tuning_curves

tokeep = np.hstack((adn, lmn))

tmp = cPickle.load(open(path2+'/figures_poster_2021/fig_cosyne_decoding.pickle', 'rb'))

decoding = {
	'wak':nap.Tsd(t=tmp['wak'].index.values, d=tmp['wak'].values, time_units = 'us'),
	'sws':nap.Tsd(t=tmp['sws'].index.values, d=tmp['sws'].values, time_units = 'us'),
	'rem':nap.Tsd(t=tmp['rem'].index.values, d=tmp['rem'].values, time_units = 'us'),	
}

tmp = cPickle.load(open(path2+'/DATA_FIG_1_ADN_LMN.pickle', 'rb'))

peaks = tmp['peaks']

###############################################################################################################
# PLOT
###############################################################################################################

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(2))

outergs = GridSpec(2,1, figure = fig, height_ratios = [0.5, 0.5], hspace = 0.4)

#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,3, 
	subplot_spec = outergs[0,0], width_ratios = [0.22, 0.1, 0.4],
	wspace = 0.3)

names = ['ADN', 'LMN']

#####################################
# Histo
#####################################
gs_histo = gridspec.GridSpecFromSubplotSpec(2,2, 
	subplot_spec = gs1[0,0], height_ratios=[0.6,0.4],
	wspace = 0.01, hspace = 0.4)

subplot(gs_histo[0,:])
noaxis(gca())
img = mpimg.imread('/home/guillaume/Dropbox/CosyneData/brain_render1.png')
imshow(img, aspect='equal')
xticks([])
yticks([])

subplot(gs_histo[1,0])
noaxis(gca())
img = mpimg.imread('/home/guillaume/Dropbox/CosyneData/histo_adn.png')
imshow(img[:, :, 0], aspect='equal', cmap='viridis')
title("ADN")
xticks([])
yticks([])

subplot(gs_histo[1,1])
noaxis(gca())
img = mpimg.imread('/home/guillaume/Dropbox/CosyneData/histo_lmn.png')
imshow(img[:,:,0], aspect='equal', cmap = 'viridis')
title("LMN")
xticks([])
yticks([])


#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gs1[0,1])

for i, st in enumerate([adn, lmn]):
	gs_tc2 = gridspec.GridSpecFromSubplotSpec(len(st),1, subplot_spec = gs_tc[i,0], hspace=0.4)
	for j, n in enumerate(peaks[st].sort_values().index.values[::-1]):
		subplot(gs_tc2[j,0])
		simpleaxis(gca())		
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,90,65])
		# clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
		tmp = tcurves[n]
		tmp = tmp/tmp.max()
		fill_between(tmp.index.values,
			np.zeros_like(tmp.index.values),
			tmp.values,
			color = clr
			)
		xticks([])
		yticks([])
		xlim(0, 2*np.pi)
		ylim(0, 1.3)
		if j == len(st)//2:
			ylabel(names[i], labelpad=20, rotation = 0)
		if j == 0:
			ylabel(str(len(st)), rotation = 0, labelpad = 8)

	if i == 1: xticks([0, 2*np.pi], [0, 360])


#########################
# RASTER PLOTS
#########################
gs_raster = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec = gs1[0,2],  hspace = 0.2)
exs = { 'wak':nap.IntervalSet(start = 7590.0, end = 7600.0, time_units='s'),
		'rem':nap.IntervalSet(start = 15710.150000, end= 15720.363258, time_units = 's'),
		'sws':nap.IntervalSet(start = 4400600.000, end = 4402154.216186978, time_units = 'ms')}

mks = 2
alp = 1
medw = 0.8

epochs = ['Wake', 'REM sleep', 'nREM sleep']

for i, ep in enumerate(exs.keys()):
	for j, st in enumerate([adn, lmn]):
		subplot(gs_raster[j,i])
		simpleaxis(gca())
		gca().spines['bottom'].set_visible(False)
		if i > 0: gca().spines['left'].set_visible(False)
		for k, n in enumerate(st):
			spk = spikes[n].restrict(exs[ep]).index.values
			if len(spk):
				clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,90,65])
				#clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.7])
				plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = medw, alpha = 0.5)
		
		ylim(0, 2*np.pi)
		xlim(exs[ep].loc[0,'start'], exs[ep].loc[0,'end'])
		xticks([])
		gca().spines['bottom'].set_visible(False)

		if i == 0: 
			yticks([0, 2*np.pi], ["0", "360"])			
		else:
			yticks([])
		
		if ep == 'wak':
			tmp = position['ry'].restrict(exs[ep])
			tmp	= tmp.as_series().rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)	
			plot(tmp, linewidth = 1, color = (0.4, 0.4, 0.4), label = 'Head-direction')
			tmp2 = decoding['wak']
			tmp2 = nap.Tsd(tmp2, time_support = wake_ep)
			tmp2 = smoothAngle(tmp2, 1)
			tmp2 = tmp2.restrict(exs[ep])
			plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp) 
			if j == 1:
				legend(frameon=False, handlelength = 1, bbox_to_anchor=(1,-0.1))

		if ep == 'rem':
			tmp2 = decoding['rem'].restrict(exs[ep])
			plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp, label = 'Decoded head-direction')
			if j == 1:
				legend(frameon=False, handlelength = 2, bbox_to_anchor=(2,-0.1))

		if ep == 'sws':
			tmp2 = decoding['sws']
			tmp3 = pd.Series(index = tmp2.index, data = np.unwrap(tmp2.values)).rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
			tmp3 = tmp3%(2*np.pi)
			tmp2 = nap.Tsd(tmp3).restrict(exs[ep])
			plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 1, color = 'gray', alpha = alp)
			plot(tmp2.loc[tmp2.idxmax()+0.03:],'--', linewidth = 1, color = 'gray', alpha = alp)


		if i == 0 and j == 1:			
			plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
			xlabel('1s', horizontalalignment='right', x=1.0)
		if i == 1 and j == 1:			
			plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
			xlabel('1s', horizontalalignment='right', x=1.0)
		if i == 2 and j == 1:			
			plot(np.array([exs[ep].end[0]-0.2, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
			xlabel('0.2s', horizontalalignment='right', x=1.0)

		if j == 0:
			title(epochs[i], pad = -1)



###############################
# Correlation
###############################
gs2 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[1,0],
	width_ratios = [0.3, 0.2])#, wspace = 0.3)#, hspace = 0.5)

gscor = gridspec.GridSpecFromSubplotSpec(2,4, subplot_spec = gs2[0,0], 
	wspace = 1, hspace = 0.6, width_ratios=[0.08, 0.5, 0.5, 0.5])

#gscor2 = gridspec.GridSpecFromSubplotSpec(4,2, subplot_spec = gs2[0,1], height_ratios=[0.1, 0.1, 0.6, 0.2], hspace=0.01)

allaxis = []


paths = [path2+'/All_correlation_ADN.pickle',
	#path2+'/All_correlation_ADN_LMN.pickle',
	path2+'/All_correlation_LMN.pickle'
]
#names = ['ADN', 'ADN/LMN', 'LMN']
#clrs = ['lightgray', 'darkgray', 'gray']
#clrs = ['sandybrown', 'olive']
clrs = ['lightgray', 'gray']
names = ['ADN', 'LMN']

mkrs = 6

xpos = [0, 2]

for i, (p, n) in enumerate(zip(paths, names)):
	# 
	data3 = cPickle.load(open(p, 'rb'))
	allr = data3['allr']
	# pearsonr = data3['pearsonr']
	print(n, allr.shape)
	print(len(np.unique(np.array([[p[0].split('-')[0], p[1].split('-')[0]] for p in np.array(allr.index.values)]).flatten())))

	#############################################################
	subplot(gscor[i, 1])
	simpleaxis(gca())
	scatter(allr['wak'], allr['rem'], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0, s = mkrs)
	m, b = np.polyfit(allr['wak'].values, allr['rem'].values, 1)
	x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
	r, p = scipy.stats.pearsonr(allr['wak'], allr['rem'])
	plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)), linewidth = 1)
	if i == 1: xlabel('Wake corr. (r)')
	ylabel('REM corr. (r)')	
	text(-1, 0.5, n, horizontalalignment='center', verticalalignment='center', transform=gca().transAxes, fontsize = fontsize)	
	legend(handlelength = 0.0, loc='center', bbox_to_anchor=(0.15, 0.9, 0.5, 0.5), framealpha =0)
	ax = gca()
	# ax.set_aspect(1)
	locator_params(axis='y', nbins=3)
	locator_params(axis='x', nbins=3)
	#xlabel('Wake corr. (r)')
	allaxis.append(gca())

	#############################################################
	subplot(gscor[i, 2])
	simpleaxis(gca())
	scatter(allr['wak'], allr['sws'], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0, s = mkrs)
	m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
	x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
	r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
	plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)), linewidth = 1)
	if i == 1: xlabel('Wake corr. (r)')
	ylabel('nREM corr. (r)')
	legend(handlelength = 0.0, loc='center', bbox_to_anchor=(0.15, 0.9, 0.5, 0.5), framealpha =0)
	# title(n, pad = 12)
	ax = gca()
	aspectratio=1.0
	ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
	#ax.set_aspect(ratio_default*aspectratio)
	# ax.set_aspect(1)
	locator_params(axis='y', nbins=3)
	locator_params(axis='x', nbins=3)
	allaxis.append(gca())

	#############################################################
	subplot(gscor[i,3])
	simpleaxis(gca())	
	marfcol = [clrs[i], 'white']

	y = data3['pearsonr']
	y = y[y['count'] > 6]
	for j, e in enumerate(['rem', 'sws']):
		plot(np.ones(len(y))*j + np.random.randn(len(y))*0.1, y[e], 'o',
			markersize=2, markeredgecolor = clrs[i], markerfacecolor = marfcol[j])
		plot([j-0.2, j+0.2], [y[e].mean(), y[e].mean()], '-', color = 'red')
	xticks([0, 1])
	xlim(-0.4,1.4)
	# print(scipy.stats.ttest_ind(y["rem"], y["sws"]))
	print(scipy.stats.wilcoxon(y["rem"], y["sws"]))

	ylim(0, 1.3)
	yticks([0, 1], [0, 1])
	# title(names[i], pad = 12)

	gca().spines.left.set_bounds((0, 1))
	ylabel("r")

	if i == 0:
		xticks([0,1], ['', ''])
	else:
		xticks([0, 1], ['REM', 'nREM'])
		text(0.5, -0.45, 'vs Wake', horizontalalignment='center', verticalalignment='center',
			transform=gca().transAxes
			)

	lwdtt = 0.1
	plot([0,1],[1.2,1.2], '-', color = COLOR, linewidth=lwdtt)
	plot([0,0],[1.15,1.2], '-', color = COLOR, linewidth=lwdtt)
	plot([1,1],[1.15,1.2], '-', color = COLOR, linewidth=lwdtt)
	TXT = ['n.s.', 'p<0.001']
	text(0.5, 1.4, TXT[i], 
		horizontalalignment='center', verticalalignment='center',
		)	



xlims = []
ylims = []
for ax in allaxis:
	xlims.append(ax.get_xlim())
	ylims.append(ax.get_ylim())
xlims = np.array(xlims)
ylims = np.array(ylims)
xl = (np.min(xlims[:,0]), np.max(xlims[:,1]))
yl = (np.min(ylims[:,0]), np.max(ylims[:,1]))
for ax in allaxis:
	ax.set_xlim(xl)
	ax.set_ylim(yl)


###############################
# MUA CROSS CORR
###############################
gsmua = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = gs2[0,1])





outergs.update(top= 0.97, bottom = 0.09, right = 0.96, left = 0.025)


savefig("/home/guillaume/LMNphysio/figures/figures_paper_2023/fig1.pdf", dpi = 200, facecolor = 'white')
#show() 
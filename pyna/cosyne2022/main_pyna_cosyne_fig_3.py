# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 17:49:50
# @Last Modified by:   gviejo
# @Last Modified time: 2022-03-16 11:41:26
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
	fig_height = fig_width*golden_mean*1.5         # height in inches
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

outergs = gridspec.GridSpec(2, 2, figure=fig, height_ratios = [0.4, 0.4], wspace = 0.3, hspace = 0.3)

names = ['ADN', 'LMN']
clrs = ['sandybrown', 'olive']

Epochs = ['Wake', 'non-REM sleep']
Epochs2 = ['Wake', 'non-REM']

path2 = '/home/guillaume/Dropbox/CosyneData'

#################################################################################################
########################################################################################
# EXAMPLE ISI
########################################################################################
#gs = gridspec.GridSpecFromSubplotSpec(1,2, outergs[0,0], width_ratios=[0.65,0.5], hspace = 0.2, wspace = 0.3)
name = 'A5011-201014A'
path = '/home/guillaume/Dropbox/CosyneData/A5011-201014A'
data = nap.load_session(path, 'neurosuite')
spikes = data.spikes
angle = data.position['ry']
position = data.position
wake_ep = data.epochs['wake']
sleep_ep = data.epochs['sleep']
sws_ep = data.read_neuroscope_intervals('sws')
rem_ep = data.read_neuroscope_intervals('rem')
wake_ep = wake_ep.loc[[0]]
tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)
SI = nap.compute_1d_mutual_info(tuning_curves, angle, angle.time_support.loc[[0]], minmax=(0,2*np.pi))
spikes.set_info(SI=SI)
spikes = spikes.getby_threshold('SI', 0.1, op = '>')
tuning_curves = tuning_curves[spikes.keys()]
tokeep = list(spikes.keys())
adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values
tcurves = tuning_curves
mks = 7
alp = 0.6
gs1 = gridspec.GridSpecFromSubplotSpec(3,2, outergs[0,0], hspace = 0.2, wspace = 0.2)



exs = {'wak':nap.IntervalSet(start = 7587976595.668784, end = 7604189853.273991, time_units='us'),
		'sws':nap.IntervalSet(start = 15038.3265, end = 15039.4262, time_units = 's')}

neurons={'adn':adn,'lmn':lmn}
decoding = cPickle.load(open(os.path.join(path2, 'figures_poster_2022/fig_cosyne_decoding.pickle'), 'rb'))

peak = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

n_adn = peak[adn].sort_values().index.values[-1]
n_lmn = peak[lmn].sort_values().index.values[-10]

ex_neurons = [n_adn, n_lmn]

for j, e in enumerate(['wak', 'sws']):
	subplot(gs1[0,j])
	simpleaxis(gca())
	for i, st in enumerate(['adn', 'lmn']):
		if e == 'wak':
			angle2 = angle
		if e == 'sws':
			angle2 = decoding['sws']

		spk = spikes[ex_neurons[i]]
		isi = nap.Tsd(t = spk.index.values[0:-1], d=np.diff(spk.index.values))
		idx = angle2.index.get_indexer(isi.index, method="nearest")
		isi_angle = pd.Series(index = angle2.index.values, data = np.nan)
		isi_angle.loc[angle2.index.values[idx]] = isi.values
		isi_angle = isi_angle.fillna(method='ffill')

		isi_angle = nap.Tsd(isi_angle)
		isi_angle = isi_angle.restrict(exs[e])

		# isi_angle = isi_angle.value_from(isi, exs[e])
		plot(isi_angle, '.-', color = clrs[i], linewidth = 2, markersize = 1)
		xlim(exs[e].loc[0,'start'], exs[e].loc[0,'end'])
	xticks([])
	title(Epochs[j])
	if j == 0: ylabel('ISI (s)')

for i, st in enumerate(['adn', 'lmn']):
	for j, e in enumerate(['wak', 'sws']):
		subplot(gs1[i+1,j])	
		simpleaxis(gca())
		ylim(0, 2*np.pi)
		xticks([])
		if e == 'wak':
			tmp = position['ry'].restrict(exs[e])
			tmp	= tmp.as_series().rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
			plot(tmp, linewidth = 1.5, color = 'black', label = 'Head-direction')
		if e == 'sws':
			tmp2 = decoding['sws']
			tmp2 = nap.Tsd(tmp2, time_support = sws_ep)
			tmp2 = smoothAngle(tmp2, 1)
			tmp2 = tmp2.restrict(exs[e])
			#plot(tmp2, '--', linewidth = 1.5, color = 'black', alpha = alp, 
			plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 2, color = 'black', alpha = alp, label = 'Decoded\nhead-direction')
			plot(tmp2.loc[tmp2.idxmax()+0.01:],'--', linewidth = 2, color = 'black', alpha = alp)


		n = ex_neurons[i]
		spk = spikes[n].restrict(exs[e]).index.values	
		#clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clrs[i], markersize = mks, markeredgewidth = 1.2)
		yticks([])
		xlim(exs[e].loc[0,'start'], exs[e].loc[0,'end'])
		if i == 1 and j == 0:
			xlabel(str(int(exs[e].tot_length('s')))+' s', horizontalalignment='right', x=1.0, fontsize = 15)
		if i == 1 and j == 1:
			xlabel(str(int(exs[e].tot_length('s')))+' s', horizontalalignment='right', x=1.0, fontsize = 15)
		if j == 0:
			yticks([0, 2*np.pi], [0, 360])
			ylabel(names[i], labelpad = -12)
		if i == 1:
			legend(handlelength = 0.8, frameon=False, bbox_to_anchor=(0.4, -0.6, 0.5, 0.5), fontsize = 16)


########################################################################################
# LONG TERM ISI
########################################################################################




data = cPickle.load(open(os.path.join(path2, 'ALL_LOG_ISI.pickle'), 'rb'))
logisi = data['logisi']
frs = data['frs']

cmaps = ['viridis', 'viridis']
mkrstype = ['-', '--']

gs = gridspec.GridSpecFromSubplotSpec(3,2, outergs[0,1], hspace=0.3, wspace = 0.3)

for i, st in enumerate(['adn', 'lmn']):	
	for j, e in enumerate(['wak', 'sws']):
		gs2 = gridspec.GridSpecFromSubplotSpec(1,2,gs[i+1,j], width_ratios=[0.2,0.8])
		tmp = logisi[st][e]
		extents = [tmp.index[0], tmp.index[-1], 0, tmp.shape[1]]
		order = frs[st][e].sort_values().index.values[::-1]
		bins = tmp.index.values
		####################################
		subplot(gs2[0,0])
		simpleaxis(gca())
		plot(frs[st][e][order].values, np.arange(len(order)), color = clrs[i], linewidth = 3)
		yticks([len(order)])
		ylim(0, len(order))
		if j == 0: ylabel(names[i], labelpad = -15)
		if i == 1: xlabel("Fr. (Hz)")			
		
		####################################
		subplot(gs2[0,1])
		tmp = tmp[order]
		tmp = tmp/tmp.max()
		tmp = gaussian_filter(tmp, sigma=(2,1))
		
		imshow(tmp.T, 
			#extent = extents,
			origin = 'lower', 
			aspect = 'auto', 
			cmap = 'jet')
		
		xt = [np.argmin(np.abs(bins - x)) for x in [10**-2, 1, 10**2]]		
		yticks([])
		if i == 0: 
			xticks([])
		if i == 1:
			xlabel("ISI (s)")		
			xticks(xt, ['$10^{-2}$', '$10^0$', '$10^2$'])

for j, e in enumerate(['wak', 'sws']):
	gs2 = gridspec.GridSpecFromSubplotSpec(1,2,gs[0,j], width_ratios=[0.2,0.8])
	subplot(gs2[0,1])
	simpleaxis(gca())	
	for i, st in enumerate(['adn', 'lmn']):
		tmp = logisi[st][e]		
		#tmp = tmp/tmp.max()
		m = tmp.mean(1)
		s = tmp.std(1)
		semilogx(m.index.values, m.values, label = Epochs2[j], color = clrs[i], linewidth = 3)
		fill_between(m.index.values, m.values -s.values, m.values + s.values, color = clrs[i], alpha = 0.2)
	ylabel("%")
	yticks([0, 0.02], [0, 2])	
	ylim(0,)
	# xlim(0,)
	title(Epochs2[j])
	xticks([10**-2, 10**0, 10**2], ['$10^{-2}$', '$10^0$', '$10^2$'])
	if i == 0:
		legend(handlelength = 1.5, frameon=False, bbox_to_anchor=(0.4, 0.8, 0.5, 0.5))
		

	#if i == 1:
		#xlabel("ISI (s)")







########################################################################################
# ISI HD MAPS
########################################################################################
gs1 = gridspec.GridSpecFromSubplotSpec(3,3, outergs[1,:], wspace = 0.4, hspace = 0.05, height_ratios=[0.12,0.2,0.2])

pisi = {'adn':cPickle.load(open(os.path.join(path2, 'PISI_ADN.pickle'), 'rb')),
		'lmn':cPickle.load(open(os.path.join(path2, 'PISI_LMN.pickle'), 'rb'))}


for j, e in enumerate(['wak', 'sws']):
	subplot(gs1[0,j])
	simpleaxis(gca())
	for i, st in enumerate(['adn', 'lmn']):
		tc = pisi[st]['tc_'+e]
		tc = tc/tc.max(0)
		m = tc.mean(1)
		s = tc.std(1)
		plot(m, label = names[i], color = clrs[i], linewidth = 3)
		fill_between(m.index.values,  m-s, m+s, color = clrs[i], alpha = 0.1)
		yticks([0, 0.5, 1], [0, 50, 100])
		ylim(0, 1)
		xticks([])
		xlim(-np.pi, np.pi)
	if j==0:
		ylabel(r"% rate")
	if j==1:
		legend(handlelength = 0.6, frameon=False, bbox_to_anchor=(1.2, 0.55, 0.5, 0.5))


for i, st in enumerate(['adn', 'lmn']):

	pisihd = []

	for j, e in enumerate(['wak', 'sws']):
		subplot(gs1[i+1,j])
		bins = pisi[st]['bins']
		xt = [np.argmin(np.abs(bins - x)) for x in [10**-2, 1]]
		tmp = pisi[st][e].mean(0)
		tmp2 = np.hstack((tmp, tmp, tmp))
		tmp2 = gaussian_filter(tmp2, sigma=(1,1))
		tmp3 = tmp2[:,tmp.shape[1]:tmp.shape[1]*2]		
		imshow(tmp3, cmap = 'jet', aspect= 'auto')
		xticks([0, tmp3.shape[1]//2, tmp3.shape[1]-1], ['-180', '0', '180'])
		yticks(xt, ['',''])
		if i == 0:
			xticks([])			
		if i == 1:
			xlabel('Centered HD')
		if j == 0:			
			yticks(xt, ['$10^{-2}$', '$10^0$'])
			ylabel(names[i])
		if j == 1:
			ylabel('ISI (s)')
		tmp4 = tmp3.mean(1)
		tmp4 = tmp4/tmp4.sum()
		pisihd.append(tmp4)

	for j, e in enumerate(['wak', 'sws']):
		subplot(gs1[i+1,2])
		simpleaxis(gca())
		semilogy(pisihd[j], bins[0:-1], mkrstype[j], label = Epochs2[j], color = clrs[i], linewidth = 3)		
		gca().set_ylim(bins[-1], bins[0])
		if i == 0:
			xticks([])
		if i == 1:
			xlabel("%")
		if i == 0:
			legend(handlelength = 1.5, frameon=False, bbox_to_anchor=(0.55, 1.2, 0.5, 0.5))
		yticks([10**-2, 10**0])



outergs.update(top= 0.95, bottom = 0.06, right = 0.98, left = 0.1)

#show()
savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2022 poster/figures/fig3.pdf", dpi = 100, facecolor = 'white')



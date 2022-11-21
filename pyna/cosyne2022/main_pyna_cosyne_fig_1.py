# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   gviejo
# @Last Modified time: 2022-11-19 21:56:00
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
sys.path.append('../../python')
import neuroseries as nts


def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*2.5         # height in inches
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


tuning_curves = nap.compute_1d_tuning_curves(spikes, angle, 120, minmax=(0, 2*np.pi))
tuning_curves = smoothAngularTuningCurves(tuning_curves)
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


tmp = cPickle.load(open(path2+'/figures_poster_2022/fig_cosyne_decoding.pickle', 'rb'))
peaks = tmp['peaks']

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

outergs = GridSpec(3,1, figure = fig, height_ratios = [0.4, 0.5, 0.4], hspace = 0.3)

#################################################################################################################################
#########################
# TUNING CURVes
#########################
gs1 = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = outergs[0,0], height_ratios = [0.15, 0.1], wspace = 0.3)

# ADN ###################
gs_left = gridspec.GridSpecFromSubplotSpec(3,9, subplot_spec = gs1[0,:], hspace = 0.1, wspace = 0.05)
bad = np.array([[0,0,0,1,1,1,2,2],[0,1,2,0,1,2,0,1]]).T
for i, n in enumerate(peaks[adn].sort_values().index.values[::-1]):
	#subplot(gs_left[len(adn)-i-1,0])
	subplot(gs_left[bad[i,0],bad[i,1]], projection = 'polar')
	clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	xlim(0, 2*np.pi)
	yticks([])
	fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n].index.values), tcurves[n].values, color = clr, alpha = 0.8, linewidth =0, zorder=2)	
	if i == 0: 
		tax1 = gca()

subplot(gs_left[0,0])
gca().text(0.8, 1.25, "ADN", transform = gca().transAxes, fontsize = 20)
#noaxis(gca())

# LMN ###################
#gs_left = gridspec.GridSpecFromSubplotSpec(5,3, subplot_spec = gs1[0,1], hspace = 0.2, wspace = 0.05)
bad = np.array([[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],[4,5,6,7,8,4,5,6,7,8,4,5,6,7,8]]).T
for i, n in enumerate(peaks[lmn[0:-1]].sort_values().index.values[::-1]):
	subplot(gs_left[bad[i,0],bad[i,1]], projection = 'polar')
	clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	xlim(0, 2*np.pi)
	yticks([])
	fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n].index.values), tcurves[n].values, color = clr, alpha = 0.8, linewidth =0, zorder=2)	
	if i == 0: 
		tax2 = gca()


subplot(gs_left[0,4])
gca().text(0.8, 1.25, "LMN", transform = gca().transAxes, fontsize=20)


clrs = ['sandybrown', 'darkgray', 'olive']
clrs2 = ['sandybrown', 'olive']
###########################
# FIRING RATE CORRELATION
frs = cPickle.load(open(os.path.join(path2, 'All_FR_ADN_LMN.pickle'), 'rb'))
count = 0
gs_fr = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec = gs1[1,:], hspace = 0.1, wspace = 0.5)	
for i, st in enumerate(['adn', 'lmn']):	
	for j, pairs in enumerate([('wak','rem'), ('wak', 'sws')]):
		subplot(gs_fr[0,count])
		#subplot(gs1[1,count])
		count+=1
		x, y = (frs[st][pairs[0]].values.astype(np.float), frs[st][pairs[1]].values.astype(np.float))
		scatter(x, y, color = clrs2[i])
		m, b = np.polyfit(x, y, 1)
		xx = np.linspace(x.min(), x.max(), 5)
		r, p = scipy.stats.pearsonr(x, y)
		plot(xx, xx*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)))
		legend(handlelength = 0.3, frameon=False, bbox_to_anchor=(0.3, 0.8, 0.5, 0.5))
		if i == 1 and j == 0:
			ylim(0, 50)
		ax = gca()
		aspectratio=1.0
		ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
		ax.set_aspect(ratio_default*aspectratio)
		locator_params(axis='y', nbins=3)
		locator_params(axis='x', nbins=3)
		#if i == 1 or i == 3:
		if j == 0 or j==2:				
			gca().text(0.8, -0.35, 'Wake rate (Hz)', transform=gca().transAxes)
			# if j == 0:
				#gca().text(-0.35, 1, 'REM fr. (Hz)', transform=gca().transAxes, rotation='vertical')
		if j == 0 or j==2:
			ylabel('REM rate (Hz)')
			# if j == 1:
				#gca().text(-0.35, 1, 'non-REM fr. (Hz)', transform=gca().transAxes, rotation='vertical')
		if j==1 or j==3:
			ylabel('non-REM\nrate (Hz)')

#################################################################################################################################
#########################
# RASTER PLOTS
#########################


#####################################################################################################
gs2 = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec = outergs[1,0],  hspace = 0.1)

####################################################################
# A WAKE
####################################################################
# ex_wake = nts.IntervalSet(start = 5.06792e+09, end = 5.10974e+09)
ex_wake = nap.IntervalSet(start = 7587976595.668784, end = 7604189853.273991, time_units='us')

mks = 7
ep = ex_wake
alp = 0.6
subplot(gs2[0,0])
simpleaxis(gca())

for k, n in enumerate(adn):
	spk = spikes[n].restrict(ex_wake).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)
tmp = position['ry'].restrict(ep)
tmp	= tmp.as_series().rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)	
plot(tmp, linewidth = 2, color = 'black')
tmp2 = decoding['wak']
tmp2 = nap.Tsd(tmp2, time_support = wake_ep)
tmp2 = smoothAngle(tmp2, 1)
tmp2 = tmp2.restrict(ex_wake)
plot(tmp2, '--', linewidth = 2, color = 'black', alpha = alp) 

ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([0, 2*np.pi], ["0", "360"])
#xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
xticks([])
ylabel("ADN", labelpad = -10)
gca().spines['bottom'].set_visible(False)
#gca().text(-0.2, 1.15, 'B', transform=gca().transAxes)



subplot(gs2[1,0])
simpleaxis(gca())

for k, n in enumerate(lmn):
	spk = spikes[n].restrict(ex_wake).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)
tmp = position['ry'].restrict(ep)
tmp	= tmp.as_series().rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)
plot(tmp, linewidth = 2, color = 'black', label = 'Head-direction')
tmp2 = decoding['wak']
tmp2 = nap.Tsd(tmp2, time_support = wake_ep)
tmp2 = smoothAngle(tmp2, 1)
tmp2 = tmp2.restrict(ex_wake)
plot(tmp2, '--', linewidth = 2, color = 'black', alpha = alp) 
plot(np.array([ep.end[0]-1, ep.end[0]]), [0, 0], linewidth = 3, color = 'black')
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([0, 2*np.pi], ["0", "360"])
xticks([])
xlabel('1s', horizontalalignment='right', x=1.0)
#xticks(np.array([ep.end[0]-1e6, ep.end[0]]), ['0', str(int(ep.tot_length('s')))+' s'])
ylabel("LMN", labelpad = -10)
#title("Wake", fontsize = 1)
xticks([])
gca().spines['bottom'].set_visible(False)
legend(frameon=False, handlelength = 1, bbox_to_anchor=(1,2.2))

# ###################################################################
# # C REM
# ###################################################################
ex_rem = nap.IntervalSet(start = 15710150000, end= 15724363258, time_units = 'us')
ep = ex_rem

subplot(gs2[0,1])
noaxis(gca())

for k, n in enumerate(adn):
	spk = spikes[n].restrict(ex_rem).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)

# tmp2 = decoding['rem'].rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
# tmp2 = nts.Tsd(tmp2).restrict(ep)
tmp2 = decoding['rem'].restrict(ep)
plot(tmp2, '--', linewidth = 2, color = 'black', alpha = alp)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
xticks([])
#title("REM sleep")#, fontsize = 44)
gca().spines['bottom'].set_visible(False)


subplot(gs2[1,1])
noaxis(gca())

for k, n in enumerate(lmn):
	spk = spikes[n].restrict(ex_rem).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)

# tmp2 = decoding['rem'].rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
# tmp2 = nts.Tsd(tmp2).restrict(ep)
tmp2 = decoding['rem'].restrict(ep)
plot(tmp2, '--', linewidth = 2, color = 'black', alpha = alp, label = 'Decoded head-direction')
plot(np.array([ep.end[0]-1, ep.end[0]]), [0, 0], linewidth = 3, color = 'black')
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
xticks([])
xlabel('1s', horizontalalignment='right', x=1.0)
gca().spines['bottom'].set_visible(False)
legend(frameon=False, handlelength = 1, bbox_to_anchor=(1,2.2))

# ###################################################################
# # C SWS
# ###################################################################
#ex_sws = nts.IntervalSet(start = 4399905437.713542, end = 4403054216.186978)
ex_sws = nap.IntervalSet(start = 4400600000, end = 4403054216.186978, time_units = 'us')
ep = ex_sws
subplot(gs2[0,2])
simpleaxis(gca())

for k, n in enumerate(adn):
	spk = spikes[n].restrict(ep).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)
tmp2 = decoding['sws']
tmp3 = pd.Series(index = tmp2.index, data = np.unwrap(tmp2.values)).rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
tmp3 = tmp3%(2*np.pi)
tmp2 = nap.Tsd(tmp3).restrict(ep)
plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 2, color = 'black', alpha = alp)
plot(tmp2.loc[tmp2.idxmax()+0.03:],'--', linewidth = 2, color = 'black', alpha = alp)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([])
#xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
xticks([])
#title("non-REM sleep")#, fontsize = 44)
gca().spines['left'].set_visible(False)
gca().spines['bottom'].set_visible(False)


subplot(gs2[1,2])
simpleaxis(gca())

for k, n in enumerate(lmn):
	spk = spikes[n].restrict(ep).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)
tmp2 = decoding['sws']
tmp3 = pd.Series(index = tmp2.index, data = np.unwrap(tmp2)).rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
tmp3 = tmp3%(2*np.pi)
tmp2 = nap.Tsd(tmp3).restrict(ep)
plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 2, color = 'black', alpha = alp)
plot(tmp2.loc[tmp2.idxmax()+0.03:],'--', linewidth = 2, color = 'black', alpha = alp)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([])
#xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
xticks([])
gca().spines['left'].set_visible(False)
gca().spines['bottom'].set_visible(False)
plot(np.array([ep.end[0]-0.5, ep.end[0]]), [0, 0], linewidth = 3, color = 'black')
xlabel('0.5s', horizontalalignment='right', x=1.0)
# ylabel("HD neurons", labelpad = -10, fontsize = 10)
# title("non-REM sleep", fontsize = 1)



#################################################################################################################################
################################
# Correlation
################################
gs3 = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec = outergs[2,0], wspace = 0.1)
gscor = gs3
# gscor = gridspec.GridSpecFromSubplotSpec(3,2, subplot_spec = gs3[0,0], wspace = 0.2)

wakeremaxis = []
wakeswsaxis = []

paths = [path2+'/All_correlation_ADN.pickle',
	path2+'/All_correlation_ADN_LMN.pickle',
	path2+'/All_correlation.pickle'
]
names = ['ADN', 'ADN/LMN', 'LMN']
#clrs = ['lightgray', 'darkgray', 'gray']


for i, (p, n) in enumerate(zip(paths, names)):
	# 

	data3 = cPickle.load(open(p, 'rb'))
	allr = data3['allr']

	print(n, allr.shape)
	print(len(np.unique(np.array([[p[0].split('-')[0], p[1].split('-')[0]] for p in np.array(allr.index.values)]).flatten())))

	subplot(gscor[0,i])
	scatter(allr['wak'], allr['rem'], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0)
	m, b = np.polyfit(allr['wak'].values, allr['rem'].values, 1)
	x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
	r, p = scipy.stats.pearsonr(allr['wak'], allr['rem'])
	plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)))
	if i == 0: ylabel('REM corr. (r)')
	title(n)
	#text(-0.6, 0.5, n, horizontalalignment='center', verticalalignment='center', transform=gca().transAxes, fontsize = 21)	
	#legend(handlelength = 0.3)
	legend(handlelength = 0.3, bbox_to_anchor=(0.7, -0.1, 0.5, 0.5))
	ax = gca()
	aspectratio=1.0
	ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
	ax.set_aspect(ratio_default*aspectratio)
	locator_params(axis='y', nbins=3)
	locator_params(axis='x', nbins=3)
	# if i == 2: xlabel('Wake corr. (r)')
	wakeremaxis.append(gca())
	# if i == 1:
	# 	text(0.5, 1.2, 'Pairwise correlation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

	# #
	subplot(gscor[1,i])
	scatter(allr['wak'], allr['sws'], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0)
	m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
	x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
	r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
	plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)))
	xlabel('Wake corr. (r)')
	if i == 0: ylabel('non-REM corr. (r)')
	legend(handlelength = 0.3, bbox_to_anchor=(0.7, -0.1, 0.5, 0.5))
	ax = gca()
	aspectratio=1.0
	ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
	ax.set_aspect(ratio_default*aspectratio)
	locator_params(axis='y', nbins=3)
	locator_params(axis='x', nbins=3)
	wakeswsaxis.append(gca())


for l in [wakeremaxis, wakeswsaxis]:
	xlims = []
	ylims = []
	for ax in l:
		xlims.append(ax.get_xlim())
		ylims.append(ax.get_ylim())
	xlims = np.array(xlims)
	ylims = np.array(ylims)
	xl = (np.min(xlims[:,0]), np.max(xlims[:,1]))
	yl = (np.min(ylims[:,0]), np.max(ylims[:,1]))
	for ax in l:
		ax.set_xlim(xl)
		ax.set_ylim(yl)



# tax1.text(-0.2, 1.3, 'A', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21, weight="bold")		
# tax1.text(3.6, 1.3, 'B', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21 , weight="bold")		
# tax1.text(15.0, 1.3, 'E', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21, weight="bold")		

# tax2.text(-0.2, 1.3, 'C', horizontalalignment='center', verticalalignment='center', transform=tax2.transAxes, fontsize = 21, weight="bold")		
# tax2.text(3.6, 1.3, 'D', horizontalalignment='center', verticalalignment='center', transform=tax2.transAxes, fontsize = 21 , weight="bold")		



outergs.update(top= 0.97, bottom = 0.04, right = 0.96, left = 0.06)

savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2022 poster/figures/fig1.pdf", dpi = 200, facecolor = 'white')
#show()
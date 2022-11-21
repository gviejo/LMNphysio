# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-03 14:52:09
# @Last Modified by:   gviejo
# @Last Modified time: 2022-11-20 23:20:11
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
	fig_width = 7
	fig_height = fig_width*golden_mean*scale         # height in inches
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

fontsize = 7

rcParams['font.family'] = 'Helvetica'
rcParams['font.size'] = fontsize
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
rcParams['axes.linewidth'] = 0.6
rcParams['axes.edgecolor'] = 'grey'
rcParams['axes.axisbelow'] = True
# rcParams['xtick.color'] = 'grey'
# rcParams['ytick.color'] = 'grey'

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

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(0.65))

outergs = GridSpec(1,1, figure = fig)#, height_ratios = [0.5, 0.4], hspace = 0.4)

#####################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec = outergs[0,0], width_ratios = [0.08, 0.3, 0.2, 0.2], wspace=0.4, hspace=0.52)

names = ['ADN', 'LMN']

#########################
# TUNING CURVes
#########################
gs_tc = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gs1[0,0])

for i, st in enumerate([adn, lmn]):
	gs_tc2 = gridspec.GridSpecFromSubplotSpec(len(st),1, subplot_spec = gs_tc[i,0])
	for j, n in enumerate(peaks[st].sort_values().index.values[::-1]):
		subplot(gs_tc2[j,0])
		simpleaxis(gca())		
		#clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
		fill_between(tcurves[n].index.values,
			np.zeros_like(tcurves[n].index.values),
			tcurves[n].values,
			color = clr
			)
		xticks([])
		yticks([])
		xlim(0, 2*np.pi)
		if j == len(st)//2:
			ylabel(names[i], labelpad=20, rotation = 0)
		if j == 0:
			ylabel(str(len(st)), rotation = 0, labelpad = 8)

	if i == 1: 
		xticks([0, 2*np.pi], [0, 360])
		xlabel("Head-direction\n(deg)")


#########################
# RASTER PLOTS
#########################
gs_raster = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = gs1[0,1],  hspace = 0.2)

exs = { 'wak':nap.IntervalSet(start = 7590.0, end = 7600.0, time_units='s'),
		# 'rem':nap.IntervalSet(start = 15710.150000, end= 15720.363258, time_units = 's'),
		'sws':nap.IntervalSet(start = 4400600.000, end = 4402154.216186978, time_units = 'ms')}

mks = 4
alp = 1
medw = 1.5

epochs = ['Wake', 'nREM sleep']

for i, ep in enumerate(exs.keys()):
	for j, st in enumerate([adn, lmn]):
		subplot(gs_raster[j,i])
		simpleaxis(gca())
		gca().spines['bottom'].set_visible(False)
		if i > 0: gca().spines['left'].set_visible(False)
		for k, n in enumerate(st):
			spk = spikes[n].restrict(exs[ep]).index.values
			if len(spk):
				#clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
				clr = hsv_to_rgb([tcurves[n].idxmax()/(2*np.pi),0.6,0.6])
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
			plot(tmp, linewidth = 1, color = 'black', label = 'Head-direction')
			tmp2 = decoding['wak']
			tmp2 = nap.Tsd(tmp2, time_support = wake_ep)
			tmp2 = smoothAngle(tmp2, 1)
			tmp2 = tmp2.restrict(exs[ep])
			plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp) 
			if j == 1:
				legend(frameon=False, handlelength = 1, bbox_to_anchor=(1,-0.14))

		# if ep == 'sws':
		# 	tmp2 = decoding['sws'].restrict(exs[ep])
		# 	plot(tmp2, '--', linewidth = 1, color = 'gray', alpha = alp, label = 'Decoded head-direction')
		# 	if j == 1:
		# 		legend(frameon=False, handlelength = 2, bbox_to_anchor=(1.5,-0.1))

		if ep == 'sws':
			tmp2 = decoding['sws']
			tmp3 = pd.Series(index = tmp2.index, data = np.unwrap(tmp2.values)).rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
			tmp3 = tmp3%(2*np.pi)
			tmp2 = nap.Tsd(tmp3).restrict(exs[ep])
			plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 1, color = 'gray', alpha = alp)
			plot(tmp2.loc[tmp2.idxmax()+0.03:],'--', linewidth = 1, color = 'gray', alpha = alp, label = 'Decoded\nhead-direction')
			if j==1: legend(frameon=False, handlelength = 2, bbox_to_anchor=(1.3,-0.08))

		if i == 0 and j == 1:			
			plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
			xlabel('1s', horizontalalignment='right', x=1.0)
		# if i == 1 and j == 1:			
		# 	plot(np.array([exs[ep].end[0]-1, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
		# 	xlabel('1s', horizontalalignment='right', x=1.0)
		if i == 1 and j == 1:			
			plot(np.array([exs[ep].end[0]-0.5, exs[ep].end[0]]), [0, 0], linewidth = 1, color = 'black')
			xlabel('0.5s', horizontalalignment='right', x=1.0)

		if j == 0:
			title(epochs[i], pad = -1)



###############################
# Correlation
###############################
# gs2 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[1,0], width_ratios = [0.5, 0.9], wspace = 0.3)#, hspace = 0.5)

# gscor = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = gs2[0,0], wspace = 0.5, hspace = 0.5)

gscor = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gs1[0,2],  hspace = 0.3)

allaxis = []


paths = [path2+'/All_correlation_ADN.pickle',
	#path2+'/All_correlation_ADN_LMN.pickle',
	path2+'/All_correlation.pickle'
]
#names = ['ADN', 'ADN/LMN', 'LMN']
#clrs = ['lightgray', 'darkgray', 'gray']
#clrs = ['sandybrown', 'olive']
clrs = ['lightgray', 'gray']
names = ['ADN', 'LMN']

mkrs = 6

for i, (p, n) in enumerate(zip(paths, names)):
	# 

	data3 = cPickle.load(open(p, 'rb'))
	allr = data3['allr']

	print(n, allr.shape)
	print(len(np.unique(np.array([[p[0].split('-')[0], p[1].split('-')[0]] for p in np.array(allr.index.values)]).flatten())))

	# #
	subplot(gscor[i,0])
	simpleaxis(gca())
	scatter(allr['wak'], allr['sws'], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0, s = mkrs)
	m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
	x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
	r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
	plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)), linewidth = 1)
	
	xlabel('Wake corr. (r)')
	ylabel('nREM corr. (r)')
	legend(handlelength = 0.4, loc='center', bbox_to_anchor=(0.2, 0.65, 0.5, 0.5), framealpha =0)
	ax = gca()
	aspectratio=1.0
	ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
	#ax.set_aspect(ratio_default*aspectratio)
	ax.set_aspect(1)
	locator_params(axis='y', nbins=3)
	locator_params(axis='x', nbins=3)
	allaxis.append(gca())

	if i == 0:
		title("Pairwise correlation", y = 0.96)


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
	ax.set_ylim(xl)







# #####################################################################################
# # Cross-correlogram
# #####################################################################################
# gscc = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[1,0], wspace = 0.25, width_ratios = [0.6, 0.4])


# gscc1 = gridspec.GridSpecFromSubplotSpec(2,6, 
# 	subplot_spec = gscc[0,0], wspace = 0.4, 
# 	width_ratios=[0.2, -0.05, 0.2, 0.05, 0.3, 0.3])

# #examples_neurons = dict(zip(names, [[0, 4], [0, 6]]))
# examples_neurons = dict(zip(names, [[0, 4], [2, 6]]))

# meanwaveforms = cPickle.load(open('/home/guillaume/Dropbox/CosyneData/A5011-201014A/MeanWaveForms.pickle', 'rb'))

# ep_names = ['Wake', 'nREM']



# for i, (n, st) in enumerate(zip(names, [adn, lmn])):
# 	neurons = peaks[st].sort_values().index.values	
	
# 	ex_neurons = [neurons[examples_neurons[n][0]],neurons[examples_neurons[n][1]]]

# 	# Waveforms
# 	subplot(gscc1[i,0])
# 	noaxis(gca())
# 	ylabel(n, rotation=0, labelpad = 15)
# 	for k, l in enumerate(ex_neurons):
# 		wave = meanwaveforms[spikes._metadata.loc[l, "group"]][l].values
# 		nch = len(wave)//32
# 		wave = wave.reshape(32, nch)
# 		for j in range(nch):
# 			plot(np.arange(0+40*k, len(wave)+40*k), 20*wave[:,j]+j*5000, linewidth = 1,
# 				color = hsv_to_rgb([tcurves[l].idxmax()/(2*np.pi),0.6,0.6]))


# 	# Tuning Curves
# 	subplot(gscc1[i,2], projection='polar')
# 	xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['','','',''])
# 	yticks([])
# 	for j in range(2):
# 		clr = hsv_to_rgb([tcurves[ex_neurons[j]].idxmax()/(2*np.pi),0.6,0.6])
# 		fill_between(tcurves[ex_neurons[j]].index.values,
# 			np.zeros_like(tcurves[ex_neurons[j]].index.values),
# 			tcurves[ex_neurons[j]].values,
# 			color = clr
# 			)
		
# 	# Cross correlograms
# 	for j, ep in enumerate([wake_ep, sws_ep]):
# 		subplot(gscc1[i,j+4])
# 		simpleaxis(gca())
# 		if j == 0:
# 			cc = nap.compute_crosscorrelogram(spikes[list(np.sort(ex_neurons))], 50, 4000, ep, time_units = 'ms', norm=False)
# 			#cc = cc.loc[-2000:2000]		
# 			fill_between(cc.index.values, np.zeros_like(cc.values.flatten()), cc.values.flatten(), color = clrs[i])			
			
# 		elif j == 1:		
# 			cc = nap.compute_crosscorrelogram(spikes[list(np.sort(ex_neurons))], 2, 400, ep, time_units = 'ms', norm=False)
# 			cc = cc.loc[-0.2:0.2]
# 			fill_between(cc.index.values*1000, np.zeros_like(cc.values.flatten()), cc.values.flatten(), color = clrs[i])			

# 		if i == 1 and j == 0:		
# 			xticks([-4, 0, 4])
# 			xlabel("cross. corr (s)")
# 			#xlim(-4,4)		
# 			ylabel("Correlation (Hz)", y = 1)
# 		elif i == 1 and j == 1:
# 			xticks([-200,0,200])
# 			#xlim(-200,200)
# 			xlabel("cross. corr (ms)")
			
# 		else:
# 			xticks([])


# 		if i == 0:
# 			title(ep_names[j])

######################################################################
# AVERAGE CROSS_CORRELOGRAM
######################################################################
gscc2 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = gs1[0,3], height_ratios=[0.3,0.3,0.1], hspace=0.3, wspace=0.35)

names = ['ADN/ADN', 'LMN/LMN']
ks = ['adn-adn', 'lmn-lmn']
clrs = ['lightgray', 'darkgray']

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

	## MEAN CC
	subplot(gscc2[i,0])
	simpleaxis(gca())
	for j,gr in enumerate([group2, group3]):
		cc = cc_sws[gr]		
		cc = cc - cc.mean(0)
		cc = cc / cc.std(0)
		cc = cc.loc[-200:200]
		m  = cc.mean(1)
		s = cc.std(1)
		plot(cc.mean(1), color = clrs[j], linewidth = 1)
		fill_between(cc.index.values, m - s, m+s, color = clrs[j], alpha = 0.2, edgecolor=None)
	#gca().spines['left'].set_position('center')	
	axvline(0, color = 'grey', alpha = 0.5)
	# if i == 2: 	
	if i in [1, 2]:
		xlabel('cross. corr. (ms)')	
		xticks([-100,0,100])
	else:
		xticks([])
	locator_params(axis='y', nbins=3)
	ylabel('z', rotation = 0)#, labelpad = 0)

	title(n, pad=0.1)
	#gca().text(-0.1, 1.0, 'z', horizontalalignment='center', verticalalignment='center', transform=gca().transAxes)
	if i == 1:	
		#######################################
		cax1 = inset_axes(gca(), "30%", "40%",					
		                   bbox_to_anchor=(0.4, -0.9, 1, 1),
		                   bbox_transform=gca().transAxes, 
		                   loc = 'lower left',	                   
		                   )
				
		plot(np.arange(len(group)), allpairs.loc[group, 'ang diff'].values, color = 'grey')
		for j,gr,ps in zip(range(2),[group2, group3],[pos2,pos3]):
			plot(ps, allpairs.loc[gr, 'ang diff'].values, '-', color = clrs[j], linewidth = 3)
		xlabel("Pairs")
		ylabel("Angular \n diff.", rotation = 0, y = 0.25, labelpad=10)
		xticks([])
		yticks([])
		simpleaxis(gca())
		yticks([0, np.pi], [0, 180])
		axhspan(0, np.deg2rad(40), color =  clrs[0], alpha = 0.3)
		axhspan(np.deg2rad(140), np.deg2rad(180), color = clrs[1], alpha = 0.3)
		#######################################
	







# # tax1.text(-0.2, 1.3, 'A', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21, weight="bold")		
# # tax1.text(3.6, 1.3, 'B', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21 , weight="bold")		
# # tax1.text(15.0, 1.3, 'E', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21, weight="bold")		

# # tax2.text(-0.2, 1.3, 'C', horizontalalignment='center', verticalalignment='center', transform=tax2.transAxes, fontsize = 21, weight="bold")		
# # tax2.text(3.6, 1.3, 'D', horizontalalignment='center', verticalalignment='center', transform=tax2.transAxes, fontsize = 21 , weight="bold")		



outergs.update(top= 0.95, bottom = 0.14, right = 0.99, left = 0.06)

savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2023 abstract submission/fig_1.pdf", dpi = 200, facecolor = 'white')
#show()
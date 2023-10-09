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
from scipy.ndimage import gaussian_filter

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


markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(2))

outergs = gridspec.GridSpec(3, 2, figure=fig, height_ratios = [0.15, 0.2, 0.2], wspace = 0.2, hspace = 0.5)


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

tmp = cPickle.load(open(path2+'/figures_poster_2022/fig_cosyne_decoding.pickle', 'rb'))
peak = tmp['peaks']



#####################################################################################
# Cross-correlogram
#####################################################################################
# gscc = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outergs[0,1], hspace = 0.7, height_ratios = [0.5, 0.4])

gscc1 = gridspec.GridSpecFromSubplotSpec(2,6, 
	subplot_spec = outergs[0,0], wspace = 0.5, hspace = 0.4, 
	width_ratios=[0.2, -0.05, 0.2, 0.1, 0.3, 0.3])

names = ['ADN', 'LMN']

examples_neurons = dict(zip(names, [[0, 4], [2, 6]]))

meanwaveforms = cPickle.load(open('/home/guillaume/Dropbox/CosyneData/A5011-201014A/MeanWaveForms.pickle', 'rb'))

ep_names = ['Wake', 'nREM']

clrs = ['lightgray', 'gray']

for i, (n, st) in enumerate(zip(names, [adn, lmn])):
	neurons = peak[st].sort_values().index.values	
	ex_neurons = [neurons[examples_neurons[n][0]],neurons[examples_neurons[n][1]]]

	# Waveforms
	subplot(gscc1[i,0])
	noaxis(gca())
	ylabel(n, rotation=0, labelpad = 15)
	for k, l in enumerate(ex_neurons):
		wave = meanwaveforms[spikes._metadata.loc[l, "group"]][l].values
		nch = len(wave)//32
		wave = wave.reshape(32, nch)
		for j in range(nch):
			plot(np.arange(0+40*k, len(wave)+40*k), 20*wave[:,j]+j*5000, linewidth = 1,
				color = hsv_to_rgb([tcurves[l].idxmax()/(2*np.pi),0.6,0.6]))


	# Tuning Curves
	subplot(gscc1[i,2], projection='polar')
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['','','',''])
	yticks([])
	for j in range(2):
		clr = hsv_to_rgb([tcurves[ex_neurons[j]].idxmax()/(2*np.pi),0.6,0.6])
		fill_between(tcurves[ex_neurons[j]].index.values,
			np.zeros_like(tcurves[ex_neurons[j]].index.values),
			tcurves[ex_neurons[j]].values,
			color = clr
			)
		
	# Cross correlograms
	for j, ep in enumerate([wake_ep, sws_ep]):
		subplot(gscc1[i,j+4])
		simpleaxis(gca())
		if j == 0:
			cc = nap.compute_crosscorrelogram(spikes[list(np.sort(ex_neurons))], 50, 4000, ep, time_units = 'ms', norm=False)
			#cc = cc.loc[-2000:2000]		
			fill_between(cc.index.values, np.zeros_like(cc.values.flatten()), cc.values.flatten(), color = clrs[i])			
			
		elif j == 1:		
			cc = nap.compute_crosscorrelogram(spikes[list(np.sort(ex_neurons))], 2, 400, ep, time_units = 'ms', norm=False)
			cc = cc.loc[-0.2:0.2]
			fill_between(cc.index.values*1000, np.zeros_like(cc.values.flatten()), cc.values.flatten(), color = clrs[i])			

		if i == 1 and j == 0:		
			xticks([-4, 0, 4])
			xlabel("cross. corr (s)")
			#xlim(-4,4)		
			ylabel("Correlation (Hz)", y = 1)
		elif i == 1 and j == 1:
			xticks([-200,0,200])
			#xlim(-200,200)
			xlabel("cross. corr (ms)")
			
		else:
			xticks([])


		if i == 0:
			title(ep_names[j])


##################################################################
# AVERAGE CROSS_CORRELOGRAM
##################################################################
gscc2 = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec = outergs[0,1])

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

	## MEAN CC
	subplot(gscc2[0,i])
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
	xticks([-100,0,100])
	xlabel('cross. corr. (ms)')	

	#yticks([])
	locator_params(axis='y', nbins=3)
	if i == 0:
		ylabel('z', rotation = 0)#, labelpad = 0)
	title(n)
	#gca().text(-0.1, 1.0, 'z', horizontalalignment='center', verticalalignment='center', transform=gca().transAxes)
	if i == 2:	
		#######################################
		cax1 = inset_axes(gca(), "20%", "30%",					
		                   bbox_to_anchor=(0.9, 0.6, 1, 1),
		                   bbox_transform=gca().transAxes, 
		                   loc = 'lower left',	                   
		                   )
				
		plot(np.arange(len(group)), allpairs.loc[group, 'ang diff'].values, color = 'grey')
		for j,gr,ps in zip(range(2),[group2, group3],[pos2,pos3]):
			plot(ps, allpairs.loc[gr, 'ang diff'].values, '-', color = clrs[j], linewidth = 3)
		if i == 2: xlabel("Pairs")
		if i == 2: ylabel("Ang. \n diff.", rotation = 0, y = 0.1)
		xticks([])
		yticks([])
		simpleaxis(gca())
		yticks([0, np.pi], [0, 180])
		axhspan(0, np.deg2rad(40), color =  clrs[0], alpha = 0.3)
		axhspan(np.deg2rad(140), np.deg2rad(180), color = clrs[1], alpha = 0.3)
		#######################################
	



########################################################################################
# EXAMPLE ISI
########################################################################################
#gs = gridspec.GridSpecFromSubplotSpec(1,2, outergs[0,0], width_ratios=[0.65,0.5], hspace = 0.2, wspace = 0.3)

epochs = ['Wake', 'REM sleep', 'nREM sleep']

mks = 2
alp = 1
medw = 0.8

clrs = ['sandybrown', 'olive']

gs1 = gridspec.GridSpecFromSubplotSpec(3,2, outergs[1,0], hspace = 0.4, wspace = 0.3)



exs = {'wak':nap.IntervalSet(start = 7587976595.668784, end = 7604189853.273991, time_units='us'),
		'sws':nap.IntervalSet(start = 15038.3265, end = 15039.4262, time_units = 's')}

neurons={'adn':adn,'lmn':lmn}
decoding = cPickle.load(open(os.path.join(path2, 'figures_poster_2022/fig_cosyne_decoding.pickle'), 'rb'))

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
		plot(isi_angle, '.-', color = clrs[i], linewidth = 1, markersize = 1)
		xlim(exs[e].loc[0,'start'], exs[e].loc[0,'end'])
	xticks([])
	# title(Epochs[j])
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
			plot(tmp, linewidth = 1, color = 'black', label = 'Head-direction')
		if e == 'sws':
			tmp2 = decoding['sws']
			tmp2 = nap.Tsd(tmp2, time_support = sws_ep)
			tmp2 = smoothAngle(tmp2, 1)
			tmp2 = tmp2.restrict(exs[e])
			#plot(tmp2, '--', linewidth = 1.5, color = 'black', alpha = alp, 
			plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 1, color = 'black', alpha = alp, label = 'Decoded head-direction')
			plot(tmp2.loc[tmp2.idxmax()+0.01:],'--', linewidth = 1, color = 'black', alpha = alp)


		n = ex_neurons[i]
		spk = spikes[n].restrict(exs[e]).index.values	
		#clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		# plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clrs[i], markersize = mks, markeredgewidth = 1.2)
		yticks([])
		xlim(exs[e].loc[0,'start'], exs[e].loc[0,'end'])
		if i == 1 and j == 0:
			xlabel(str(int(exs[e].tot_length('s')))+' s', horizontalalignment='right', x=1.0)
		if i == 1 and j == 1:
			xlabel(str(int(exs[e].tot_length('s')))+' s', horizontalalignment='right', x=1.0)
		if j == 0:
			yticks([0, 2*np.pi], [0, 360])
			ylabel(names[i], labelpad = 2, rotation = 0)
		if i == 1:
			legend(handlelength = 1, frameon=False, bbox_to_anchor=(0.4, -0.6, 0.5, 0.5))







########################################################################################
# ISI HD MAPS
########################################################################################
gs1 = gridspec.GridSpecFromSubplotSpec(3,3, outergs[1,1], 
	wspace = 0.6, hspace = 0.1, height_ratios=[0.12,0.2,0.2],
	width_ratios=[0.5,0.5,0.1])

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
		plot(m, label = names[i], color = clrs[i], linewidth = 1)
		fill_between(m.index.values,  m-s, m+s, color = clrs[i], alpha = 0.1)
		yticks([0, 0.5, 1], [0, 50, 100])
		ylim(0, 1)
		xticks([])
		xlim(-np.pi, np.pi)
	if j==0:
		ylabel(r"% rate")
	if j==1:
		legend(handlelength = 0.6, frameon=False, bbox_to_anchor=(1.2, 0.55, 0.5, 0.5))

	# title(Epochs2[j])

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
		im = imshow(tmp3, cmap = 'jet', aspect= 'auto')
		xticks([0, tmp3.shape[1]//2, tmp3.shape[1]-1], ['-180', '0', '180'])
		yticks(xt, ['',''])
		if i == 0:
			xticks([])			
		if i == 1:
			xlabel('Centered HD')
		if j == 0:			
			yticks(xt, ['$10^{-2}$', '$10^0$'])
			ylabel('ISI (s)')			
		if j == 1:
			ylabel(names[i], rotation = 0, labelpad = 15)
		tmp4 = tmp3.mean(1)
		tmp4 = tmp4/tmp4.sum()
		pisihd.append(tmp4)

# cax = fig.add_axes([0.9, 0.15, 0.01, 0.1])
# fig.colorbar(im, cax=cax, orientation='vertical')


outergs.update(top= 0.95, bottom = 0.08, right = 0.96, left = 0.05)


savefig("/home/guillaume/LMNphysio/figures/figures_paper_2023/fig4.pdf", dpi = 200, facecolor = 'white')
#show()
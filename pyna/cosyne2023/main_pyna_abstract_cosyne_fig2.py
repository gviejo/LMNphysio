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
from scipy.ndimage import gaussian_filter

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
	fig_width = 3
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


markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(1.7))

outergs = gridspec.GridSpec(4, 1, figure=fig, height_ratios = [0.5, 0.35, 0.01, 0.5], wspace = 0.5, hspace = 0.5)

names = ['ADN', 'LMN']
clrs = ['sandybrown', 'olive']

Epochs = ['Wake', 'nREM sleep']
Epochs2 = ['Wake', 'nREM']

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
spikes.set_info(SI)
spikes = spikes.getby_threshold('SI', 0.1, op = '>')
tuning_curves = tuning_curves[spikes.keys()]
tokeep = list(spikes.keys())
adn = spikes._metadata[spikes._metadata["location"] == "adn"].index.values
lmn = spikes._metadata[spikes._metadata["location"] == "lmn"].index.values
tcurves = tuning_curves

mks = 2
alp = 1
medw = 0.8
	
gs1 = gridspec.GridSpecFromSubplotSpec(2,3, outergs[0,0], hspace = 0.3, wspace = 0.4, height_ratios=[0.4, 0.6], width_ratios=[0.06, 0.6, 0.6])



exs = {'wak':nap.IntervalSet(start = 7587976595.668784, end = 7604189853.273991, time_units='us'),
		'sws':nap.IntervalSet(start = 15038.3265, end = 15039.4262, time_units = 's')}

neurons={'adn':adn,'lmn':lmn}
decoding = cPickle.load(open(os.path.join(path2, 'figures_poster_2022/fig_cosyne_decoding.pickle'), 'rb'))

peak = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))

n_adn = peak[adn].sort_values().index.values[-1]
n_lmn = peak[lmn].sort_values().index.values[-10]

ex_neurons = [n_adn, n_lmn]

for j, e in enumerate(['wak', 'sws']):
	subplot(gs1[0,j+1])
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
		plot(isi_angle, '.-', color = clrs[i], linewidth = 1, markersize = 1, label=names[i])
		xlim(exs[e].loc[0,'start'], exs[e].loc[0,'end'])
	xticks([])
	title(Epochs[j])
	if j == 0: 
		ylabel('ISI (s)', labelpad=15, rotation=0, y=0.4)
		legend(frameon=False, handlelength = 0.5, bbox_to_anchor=(0.64,0.3))

	subplot(gs1[1,j+1])	
	simpleaxis(gca())
	ylim(0, 2*np.pi)
	xticks([])
	yticks([])

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

	for i, st in enumerate(['adn', 'lmn']):
		n = ex_neurons[i]
		spk = spikes[n].restrict(exs[e]).index.values	
		#clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax()+i*0.4, '|', color = clrs[i], markersize = mks, markeredgewidth = 1.2)
		
	xlim(exs[e].loc[0,'start'], exs[e].loc[0,'end'])

	xlabel(str(int(exs[e].tot_length('s')))+' s')
	gca().xaxis.set_label_coords(0.9, -0.05)

	if j == 0:
		yticks([0, 2*np.pi], [0, 360])
		ylabel("Head\ndirection")
	# if i == 1:
	# 	legend(handlelength = 1, frameon=False, bbox_to_anchor=(0.4, -0.6, 0.5, 0.5))







########################################################################################
# ISI HD MAPS
########################################################################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,8, outergs[1,0], wspace = 0.1, hspace = 0.05,
	width_ratios=[0.06, 0.1, 0.1, 0.06, 0.1, 0.1, 0.01,0.01])

pisi = {'adn':cPickle.load(open(os.path.join(path2, 'PISI_ADN.pickle'), 'rb')),
		'lmn':cPickle.load(open(os.path.join(path2, 'PISI_LMN.pickle'), 'rb'))}


xtx = [1, 2, 4, 5]

cnt = 0

for j, e in enumerate(['wak', 'sws']):

	pisihd = []

	for i, st in enumerate(['adn', 'lmn']):	

		subplot(gs1[0,xtx[cnt]])

		bins = pisi[st]['bins']
		xt = [np.argmin(np.abs(bins - x)) for x in [10**-2, 1]]
		tmp = pisi[st][e].mean(0)
		tmp2 = np.hstack((tmp, tmp, tmp))
		tmp2 = gaussian_filter(tmp2, sigma=(1,1))
		tmp3 = tmp2[:,tmp.shape[1]:tmp.shape[1]*2]		
		im = imshow(tmp3, cmap = 'jet', aspect= 'auto')		
		xticks([tmp3.shape[1]//4, tmp3.shape[1]//2, 3*tmp3.shape[1]//4], ['-90', '0', '90'])
		yticks(xt, ['',''])
		if i == 1:
			xlabel('Centered HD')
			gca().xaxis.set_label_coords(.1, -0.3)
		if st == 'adn':
			yticks(xt, ['$10^{-2}$', '$10^0$'])
		if j == 0 and i == 0:
			ylabel('ISI (s)')

		title(names[i], pad=0)
		tmp4 = tmp3.mean(1)
		tmp4 = tmp4/tmp4.sum()
		pisihd.append(tmp4)

		
		if j == 1 and i == 1:
			cax = subplot(gs1[0,-2])
			colorbar(im, cax=cax)
			title("P", pad=1)

		cnt += 1

############################################################################
# MODEL
############################################################################
gsmodel = gridspec.GridSpecFromSubplotSpec(2,4, outergs[3,0], width_ratios=[0.02, 0.14, 0.2, 0.2], hspace = 0.4, wspace = 0.4)


# Model
IO_fr = pd.read_hdf(path2+'/IO_fr.hdf')

subplot(gsmodel[:,1])
# plot(IO_fr['lmn'], 'o-', color = clrs[1], label = 'Integrator', markersize = 1, linewidth =1)
plot(IO_fr['adn'], 'o-', color = clrs[0], label = 'Activator', markersize = 1 , linewidth =1)

# legend(handlelength = 0.8, frameon=False, bbox_to_anchor=(0.5, -0.1, 0.5, 0.5))

xlabel("Input (Hz)")
ylabel("Ouput (Hz)")
title("Activation model")


# Raster plot
subplot(gsmodel[0,2])
simpleaxis(gca())

# raster plot simulation
subplot(gsmodel[1,2])
simpleaxis(gca())

# CC

titles = ['Integrator', 'Activator']
clrs = ['#E66100', '#5D3A9B']

data = cPickle.load(open(os.path.join(path2, 'MODEL_CC.pickle'), 'rb'))

# for i, st in enumerate(['int-int', 'adn-adn']):
for i, st in enumerate(['adn-adn']):
	subplot(gsmodel[:,3])
	simpleaxis(gca())
	for j in range(2):
		m = data[st][j]['m']
		s = data[st][j]['s']
		plot(m.index.values*1000, m.values, color = clrs[j], linewidth = 1)
		fill_between(m.index.values*1000, m-s, m+s, alpha = 0.2, color = clrs[j])
	gca().spines['left'].set_position('center')
	gca().text(0.6, 0.96, 'z', horizontalalignment='center', verticalalignment='center', transform=gca().transAxes)
	xlabel('cross. corr. (ms)')	
	xticks([-100,0,100])
	#yticks([])
	locator_params(axis='y', nbins=3)


	# title("Model", pad = 13)




outergs.update(top= 0.95, bottom = 0.1, right = 0.95, left = 0.05)


savefig("/home/guillaume/Dropbox/Applications/Overleaf/Cosyne 2023 abstract submission/fig_2.pdf", dpi = 200, facecolor = 'white')
#show()
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
from mtspec import mtspec, wigner_ville_spectrum


def smoothAngle(tsd, sd):
	tmp 			= pd.Series(index = tsd.index.values, data = np.unwrap(tsd.values))	
	tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=sd)
	newtsd			= nts.Tsd(tmp2%(2*np.pi))
	return newtsd

data_directory 		= '/mnt/DataGuillaume/LMN/A1407'
# data_directory 		= '../data/A1400/A1407'
info 				= pd.read_csv(os.path.join(data_directory,'A1407.csv'), index_col = 0)

session = 'A1407-190416'


good_exs_wake = [nts.IntervalSet(start = [4.96148e+09], end = [4.99755e+09]),
				nts.IntervalSet(start = [3.96667e+09], end = [3.99714e+09]),
				nts.IntervalSet(start = [5.0872e+09], end = [5.13204e+09])
				]

good_exs_rem = [nts.IntervalSet(start = [8.94993e+09], end = [8.96471e+09])]

good_exs_sws = [nts.IntervalSet(start = [8.4855e+09], end = [8.48773e+09]),
				nts.IntervalSet(start = [8.36988e+09], end = [8.37194e+09])]


data = cPickle.load(open('../../figures/figures_poster_2019/fig_1_decoding.pickle', 'rb'))

angle_wak = data['wak']
angle_rem = data['rem']
angle_sws = data['sws']
tcurves = data['tcurves']
angle = data['angle']
peaks = data['peaks']
proba_angle_sleep = data['proba_angle_sws']
spike_counts = data['spike_counts']

proba_angle_sleep = nts.TsdFrame(t =  proba_angle_sleep.index.values*1000, d = proba_angle_sleep.values)

spike_counts = nts.TsdFrame(t = spike_counts.index.values*1000, d = spike_counts.values)

path = os.path.join(data_directory, session)
spikes, shank = loadSpikeData(path)

H = peaks/(2*np.pi)

HSV = np.vstack((H*360, np.ones_like(H)*85, np.ones_like(H)*45)).T

RGB = np.array([hsluv.hsluv_to_rgb(HSV[i]) for i in range(len(HSV))])

RGB = pd.DataFrame(index = H.index, data = RGB)

neurons = tcurves.columns.values


lfp = pd.read_hdf('../../figures/figures_poster_2019/lfp_190416.h5')
lfp = nts.Tsd(lfp)

ratio = pd.read_hdf('../../figures/figures_poster_2019/ratio2.h5')
ratio = nts.Tsd(ratio)

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

outergs = GridSpec(1,2, figure = fig, width_ratios = [0.25, 0.75], wspace = 0.3)


####################################################################
# A TUNING CURVES
####################################################################
gs_left = gridspec.GridSpecFromSubplotSpec(5,2, subplot_spec = outergs[0,0])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)

for i, n in enumerate(neurons):
	subplot(gs_left[int(i/2),i%2], projection = 'polar')
	clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	yticks([])
	fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n].index.values), tcurves[n].values, color = clr, alpha = 0.8, linewidth =0, zorder=2)	
	# if i == len(neurons)-1:
	# 	# simpleaxis(gca())
	# 	xticks([0, 2*np.pi], ["0", r"$2\pi$"], fontsize = 6)
	# 	yticks([],[])
	# 	# gca().spines['left'].set_visible(False)	
	# else:
	# 	gca().axis('off')		
	

	gca().text(0.75, 1, str(int(tcurves[n].max()))+" Hz", transform = gca().transAxes, fontsize = 6)#, fontweight='bold')
	


####################################################################
# B DECODING
####################################################################
gs_right = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = outergs[0,1], hspace = 0.2)#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)


for i, angle2, epoch, ex_ep in zip(range(3), [angle_wak, angle_rem, angle_sws], ['WAKE', 'REM', 'NREM'], [good_exs_wake, good_exs_rem, good_exs_sws]):

	if epoch is 'WAKE':
		gs2 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = gs_right[i,:])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)	
	elif epoch in ['REM', 'NREM']:
		gs2 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = gs_right[i,:], height_ratios = [0.1, 0.1, 0.8])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)	


	if epoch is 'WAKE':
		subplot(gs2[0,:])
	elif epoch in ['REM', 'NREM']:
		subplot(gs2[2,:])

	simpleaxis(gca())
	# gca().spines['bottom'].set_visible(False)
	newangle = smoothAngle(angle, 10)
	
	if epoch is 'NREM':
		newangle2 = smoothAngle(angle2, 10)
	else:
		newangle2 = smoothAngle(angle2, 4)

	plot(newangle.restrict(ex_ep[0]), color = 'red', label = "Actual HD")
	
	if epoch == 'NREM':
		count = spike_counts.restrict(ex_ep[0]).sum(1)
		tmp = newangle2.restrict(ex_ep[0])
		idx = count.index.values[np.where(count < 0.3)[0]]
		tmp.loc[idx] = np.nan
		plot(tmp, color = 'gray')
		# sys.exit()
	else:
		plot(newangle2.restrict(ex_ep[0]), color = 'gray', label = "Decoded HD")


	if i == 0:
		legend(frameon = False)


	# SPIKES
	for k, n in enumerate(neurons):
		spk = spikes[n]		
		clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])
		plot(spk.restrict(ex_ep[0]).fillna(peaks[n]), '|', color = clr, markeredgewidth = 1, markersize = 3)
	ylim(0, 2*np.pi)
	yticks([0, np.pi, 2*np.pi], ['0', r"$\pi$", r"$2\pi$"])
	ylabel(epoch)
	xlim(ex_ep[0].loc[0].values)
	if epoch == 'WAKE':
		xt = np.arange(ex_ep[0].loc[0,'start'], ex_ep[0].loc[0,'end'], 15*1e6)
		xtick = [0, 15, 30]
	elif epoch == 'REM':
		xt = np.arange(ex_ep[0].loc[0,'start'], ex_ep[0].loc[0,'end'], 5*1e6)
		xtick = [0, 5, 10]
	elif epoch == 'NREM':
		xt = np.arange(ex_ep[0].loc[0,'start'], ex_ep[0].loc[0,'end'], 1*1e6)
		xtick = [0, 1, 2]		
		xlabel("Time (s)")

	xticks(xt, xtick)#, np.arange(len(xt)))

	# sys.exit()

	if epoch in ['REM', 'NREM']:

		subplot(gs2[0,:])
		noaxis(gca())
		plot(lfp.restrict(ex_ep[0]), color = 'black', linewidth = 1)
		xlim(ex_ep[0].loc[0].values)

		subplot(gs2[1,:])
		simpleaxis(gca())

		plot(ratio.restrict(ex_ep[0]), color = 'grey', linewidth = 1)
		# ylim(-1.1, 1.1)		
		xlim(ex_ep[0].loc[0].values)
		ylabel(r"$log(\frac{\theta}{\delta})$", rotation = 0, labelpad = 12, y = 0)

		axhline(0, color = 'black', linewidth = 0.78)
		yticks([0], [0])

		gca().spines['bottom'].set_visible(False)
		gca().set_xticks([])		


		# tmp = lfp.restrict(new_ep).values
		# tmp = lfp.restrict(ex_ep[0])

		# wp = pywt.WaveletPacket(tmp, 'db2', 'symmetric', maxlevel = 4)
		# f, t, Sxx = spectrogram_lspopt(tmp, 250, c_parameter=10)
		
		# wv = wigner_ville_spectrum(tmp, 1/250, frequency_divider = 6, smoothing_filter = 'gauss');imshow(np.sqrt(abs(wv)), aspect = 'auto', cmap = 'magma');show()
		# plot(tmp)
		# noaxis(gca())
		# pcolormesh(t, f, Sxx)
		# mtspec(tmp, 1/250, 3);	

		# sh1ow()
		# sys.exit()
		# spectrum, freqs, t, im = specgram(tmp, NFFT=64, noverlap=32, Fs = 250)
		# ylim(0, np.max(np.where(freqs>30)[0]))
		# ylim(0, 30)
		# noaxis(gca())

outergs.update(top= 0.97, bottom = 0.07, right = 0.97, left = 0.02)

savefig("../../figures/figures_poster_2019/fig_poster_1.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_poster_2019/fig_poster_1.pdf &")

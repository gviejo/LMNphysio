import numpy as np
import pandas as pd
import sys
sys.path.append("../")
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
import hsluv
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
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

fig = figure(figsize = figsize(1.5))

outergs = GridSpec(1,1, figure = fig)


#################################################################################################################################
gs1 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec = outergs[0,0], height_ratios = [0.3, 0.6], wspace = 0.1, hspace = 0.3)

names = ['ADN/ADN', 'LMN/ADN', 'LMN/LMN']
ks = ['adn-adn', 'adn-lmn', 'lmn-lmn']
clrs = ['lightgray', 'gray', 'darkgray']

#xlabels = ['ADN/ADN (ms)', 'LMN/ADN (ms)', 'LMN/LMN (ms)']


acc = cPickle.load(open(os.path.join('../../data', 'All_crosscor_ADN_LMN.pickle'), 'rb'))
allpairs = acc['pairs']
cc_sws = acc['cc_sws']
tcurves = acc['tcurves']

accadn = cPickle.load(open(os.path.join('../../data', 'All_crosscor_ADN_adrien.pickle'), 'rb'))

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

	## Exemple
	idx = [group2[exn[i][0]], group3[exn[i][1]]]
	axD = subplot(gs1[0,i])
	simpleaxis(gca())
	gca().spines['left'].set_position('center')		
	inset_pos = [-0.1, 0.1]
	for j in range(2):		
		# Cross correlogram SWS			
		cc = cc_sws[idx[j]].loc[-200:200]
		cc = (cc - cc.mean())/cc.std()
		cc = cc.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)			
		plot(cc, linewidth = 2, color = clrs[j])		
	title(names[i])

	gca().text(-0.0, 1.15, letters[i], horizontalalignment='center', verticalalignment='center', transform=gca().transAxes, fontsize = 21, weight="bold")


	#xlabel(xlabels[i])
	#if i == 2: xlabel("cc. (ms)")
	gca().set_ylabel("z",  y = 0.8, rotation = 0, labelpad = 15)
	yticks([])
	xticks([-100,0,100])
	for j in range(2):
		cax1 = inset_axes(axD, "30%", "30%",
	                   bbox_to_anchor=(inset_pos[j], 0.6, 1, 1),
	                   bbox_transform=axD.transAxes, 
	                   loc = 'lower left',
	                   axes_class = matplotlib.projections.get_projection_class('polar')
	                   )
		# Tuning curves	
		tmp = tcurves[list(idx[j])]
		tmp = tmp.dropna()
		if i == 0: 
			tmp = tmp.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)			
		tmp = tmp/tmp.max()
		cax1.plot(tmp.iloc[:,0], linewidth = 2, color = clrs[j])
		cax1.plot(tmp.iloc[:,1], linewidth = 2, color = clrs[j])
		#if i == 0 : legend(frameon = False, handlelength = 0.4, bbox_to_anchor=(-0.05,1.1))
		cax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
		cax1.set_xticklabels([])
		cax1.set_yticks([])




	## MEAN CC
	subplot(gs1[1,i])
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
	ylabel('z',  y = 0.9, rotation = 0, labelpad = 15)
	
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
	





outergs.update(top= 0.93, bottom = 0.1, right = 0.98, left = 0.02)

savefig("/home/guillaume/Dropbox (Peyrache Lab)/Applications/Overleaf/Cosyne 2022 abstract submission/fig_2.pdf", dpi = 200, facecolor = 'white')




		# #######################
		# # UP DOWN MODULATION
		# gsmix = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gscor[0,2], hspace = 0.4)

		# # Rate modulation up down
		# subplot(gsmix[0,0])
		# mua = cPickle.load(open('/home/guillaume/LMNphysio/data/MUA_ADN_LMN_UP_DOWN.pickle', 'rb'))
		# simpleaxis(gca())
		# gca().spines['left'].set_position('center')

		# plot(mua['adn'].mean(1), label = 'ADN')
		# plot(mua['lmn'].mean(1), label = 'LMN')
		# legend()
		# xticks([-0.5, 0.5], ['DOWN', 'UP'])


		# # Correlation up downs
		# subplot(gsmix[1,0])

		# data3 = cPickle.load(open('/home/guillaume/LMNphysio/data/All_correlation_LMN_UP_DOWN.pickle', 'rb'))
		# allr = data3['allr']
		# #scatter(allr['wak'], allr['up'], color = 'lightgray', alpha = 0.5, edgecolor = None)
		# m, b = np.polyfit(allr['wak'].values, allr['up'].values, 1)
		# x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
		# r, p = scipy.stats.pearsonr(allr['wak'], allr['up'])
		# plot(x, x*m + b, color = 'red', label = 'UP: r = '+str(np.round(r, 2)))

		# #scatter(allr['wak'], allr['down'], color = 'lightgreen', alpha = 0.5, edgecolor = None)
		# m, b = np.polyfit(allr['wak'].values, allr['down'].values, 1)
		# x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
		# r, p = scipy.stats.pearsonr(allr['wak'], allr['down'])
		# plot(x, x*m + b, color = 'lightgreen', label = 'DOWN: r = '+str(np.round(r, 2)))
		# xlabel('Wake')
		# ylabel('DOWN/UP')

		# legend(frameon=False)
		# ax = gca()
		# aspectratio=1.0
		# ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
		# ax.set_aspect(ratio_default*aspectratio)
		# locator_params(axis='y', nbins=4)
		# locator_params(axis='x', nbins=4)







		# ################################
		# # SWRS
		# gsswr = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = gs2[0,1], height_ratios = [0.1,0.2, 0.6], hspace = 0.1)


		# s = 'LMN-ADN/A5026/A5026-210726A'

		# name = s.split('/')[-1]
		# spikes, shank 						= loadSpikeData(os.path.join(data_directory, s))
		# position 							= loadPosition(path, events, episodes)
		# wake_ep 							= loadEpoch(path, 'wake', episodes)
		# rip_ep, rip_tsd 					= loadRipples(os.path.join(data_directory, s))
		# neurons = np.where(np.sum([shank==i for i in [8,9]], 0))[0]
		# tcurves 						= computeAngularTuningCurves({n:spikes[n] for n in neurons}, position['ry'], wake_ep, 121)	
		# peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()
		# tcurves 		= tcurves[peaks.index.values]

		# tmp = nts.IntervalSet(start = [8828.000], end = [8829.000], time_units = 's')
		# ex_swr = rip_tsd.restrict(tmp)
		# ep_swr = nts.IntervalSet(start = [ex_swr.index[0]-5e5], end = [ex_swr.index[0]+5e5])

		# lmn = neurons

		# # Exemple
		# subplot(gsswr[0,0])
		# noaxis(gca())
		# file = os.path.join(data_directory, s, name+'.eeg')

		# # lfp = loadLFP(file, 80, 5, frequency=1250.0, precision='int16')
		# # lfp = lfp.restrict(ep_swr)
		# # signal = butter_bandpass_filter(lfp.values, 100, 300, 1250, order = 4)
		# # signal = nts.Tsd(t = lfp.index.values, d = signal)
		# # tosave = pd.concat([lfp.restrict(ep_swr), signal.restrict(ep_swr)], 1)
		# # tosave.columns = ['raw', 'filtered']
		# # tosave.to_hdf('../../data/ex_swr_cosyne.h5', 'swr')
		# signal = pd.read_hdf('../../data/ex_swr_cosyne.h5')
		# plot(signal['filtered'], color = 'black')
		# title('100-300 Hz', loc = 'right', fontsize = 12)
		# ylabel('CA1')
		# legend(frameon = False)
		# xlim(ep_swr.loc[0,'start'], ep_swr.loc[0,'end'])

		# subplot(gsswr[1,0])
		# simpleaxis(gca())

		# for k, n in enumerate(lmn):
		# 	spk = spikes[n].restrict(ep_swr).index.values
		# 	if len(spk):
		# 		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		# 		plot(spk, np.ones_like(spk)*peaks[n], '|', color = clr, markersize = 10, markeredgewidth = 1.2)

		# ylim(0, 2*np.pi)
		# xlim(ep_swr.loc[0,'start'], ep_swr.loc[0,'end'])
		# yticks([0, 2*np.pi], ["0", "360"])
		# xticks([])
		# ylabel("LMN")


		# gca().spines['bottom'].set_visible(False)

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
	fig_height = fig_width*golden_mean*0.75          # height in inches
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
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

# s = 'LMN/A1411/A1411-200908A'
s = 'LMN-ADN/A5011/A5011-201014A'

name = s.split('/')[-1]
path = os.path.join(data_directory, s)



############################################################################################### 
# LOADING DATA
###############################################################################################
episodes 							= infos[s.split('/')[1]].filter(like='Trial').loc[s.split('/')[2]].dropna().values
episodes[episodes != 'sleep'] 		= 'wake'
events								= list(np.where(episodes != 'sleep')[0].astype('str'))	
spikes, shank 						= loadSpikeData(path)
n_channels, fs, shank_to_channel 	= loadXML(path)
position 							= loadPosition(path, events, episodes)
wake_ep 							= loadEpoch(path, 'wake', episodes)
sleep_ep 							= loadEpoch(path, 'sleep')					
sws_ep								= loadEpoch(path, 'sws')
rem_ep								= loadEpoch(path, 'rem')

# Only taking the first wake ep
wake_ep = wake_ep.loc[[0]]

# NEURONS FROM ADN	
if 'A5011' in s:
	adn = np.where(shank <=3)[0]
	lmn = np.where(shank ==5)[0]


############################################################################################### 
# COMPUTING TUNING CURVES
###############################################################################################
tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

# CHECKING HALF EPOCHS
wake2_ep = splitWake(wake_ep)
tokeep2 = []
stats2 = []
tcurves2 = []
for i in range(2):	
	tcurves_half = computeAngularTuningCurves(spikes, position['ry'], wake2_ep.loc[[i]], 121)
	tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)
	tokeep, stat = findHDCells(tcurves_half)
	tokeep2.append(tokeep)
	stats2.append(stat)
	tcurves2.append(tcurves_half)

tokeep = np.intersect1d(tokeep2[0], tokeep2[1])


# Checking firing rate
spikes = {n:spikes[n] for n in tokeep}
mean_frate 							= computeMeanFiringRate(spikes, [wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws'])	
# tokeep = mean_frate[(mean_frate.loc[tokeep]>4).all(1)].index.values
# tokeep = mean_frate[mean_frate.loc[tokeep,'sws']>1].index.values

tcurves 		= tuning_curves[tokeep]
peaks 			= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()
tcurves 		= tcurves[peaks.index.values]

adn = np.intersect1d(adn, tokeep)
lmn = np.intersect1d(lmn, tokeep)

tokeep = np.hstack((adn, lmn))


decoding = cPickle.load(open('../../figures/figures_poster_2021/fig_cosyne_decoding.pickle', 'rb'))

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

fig = figure(figsize = figsize(3))

outergs = GridSpec(1,2, figure = fig, width_ratios = [0.6, 0.3])


#################################################################################################################################
gs1 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[0,0], width_ratios = [0.2, 0.6], wspace = 0.1)


##################
# Brain RENDER
#################
#subplot(gs1[:,0])

#########################
# TUNING CURVes
#########################
gs12 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gs1[0,0])


# ADN ###################
gs_left = gridspec.GridSpecFromSubplotSpec(4,3, subplot_spec = gs12[0,0], hspace = 0.1, wspace = 0.1)
#bad = [(0,3),(0,4),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4)]
bad = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1)]
for i, n in enumerate(peaks[adn].sort_values().index.values[::-1]):
	#subplot(gs_left[len(adn)-i-1,0])
	subplot(gs_left[bad[i]], projection = 'polar')
	clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
	gca().grid(zorder=0)
	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
	xlim(0, 2*np.pi)
	yticks([])
	fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n].index.values), tcurves[n].values, color = clr, alpha = 0.8, linewidth =0, zorder=2)	
	if i == 0: 
		tax1 = gca()

subplot(gs_left[0,0])
gca().text(0.8, 1.25, "ADN", transform = gca().transAxes)
#noaxis(gca())

# LMN ###################
gs_left = gridspec.GridSpecFromSubplotSpec(4,3, subplot_spec = gs12[1,0], hspace = 0.1, wspace = 0.1)
#bad = np.array([[0,0,0,1,1,1,1,1,2,2,2,2,2],[2,3,4,0,1,2,3,4,0,1,2,3,4]]).T
bad = np.array([[0,0,0,1,1,1,2,2,2,3,3,3],[0,1,2,0,1,2,0,1,2,0,1,2]]).T
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

subplot(gs_left[0,0])
gca().text(0.8, 1.25, "LMN", transform = gca().transAxes)


#####################################################################################################
gs13 = gridspec.GridSpecFromSubplotSpec(5,3, subplot_spec = gs1[0,1],  wspace = 0.07, height_ratios = [0.1, 0.6, 0.3, 0.6, 0.1])
####################################################################
# A WAKE
####################################################################
# ex_wake = nts.IntervalSet(start = 5.06792e+09, end = 5.10974e+09)
ex_wake = nts.IntervalSet(start = 7587976595.668784, end = 7604189853.273991)

mks = 7
ep = ex_wake
alp = 0.6
subplot(gs13[1,0])
simpleaxis(gca())

for k, n in enumerate(adn):
	spk = spikes[n].restrict(ex_wake).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)
tmp = position['ry'].restrict(ep)
tmp	= tmp.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)	
plot(tmp, linewidth = 2, color = 'black', label = 'Head-direction')
tmp2 = decoding['wak'].rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
tmp2 = nts.Tsd(tmp2).restrict(ep)
plot(tmp2, '--', linewidth = 2, color = 'black', alpha = alp) 
legend(frameon=False, handlelength = 1, bbox_to_anchor=(1.1,-0.05))
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([0, 2*np.pi], ["0", "360"])
#xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
xticks([])
#ylabel("ADN", labelpad = -10)
title("Wake")#, fontsize = 44)
gca().spines['bottom'].set_visible(False)
#gca().text(-0.2, 1.15, 'B', transform=gca().transAxes)



subplot(gs13[3,0])
simpleaxis(gca())

for k, n in enumerate(lmn):
	spk = spikes[n].restrict(ex_wake).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)
tmp = position['ry'].restrict(ep)
tmp	= tmp.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)
plot(tmp, linewidth = 2, color = 'black')
tmp2 = decoding['wak'].rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
tmp2 = nts.Tsd(tmp2).restrict(ep)
plot(tmp2, '--', linewidth = 2, color = 'black', alpha = alp) 
plot(np.array([ep.end[0]-1e6, ep.end[0]]), [0, 0], linewidth = 3, color = 'black')
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([0, 2*np.pi], ["0", "360"])
xticks([])
xlabel('1s', horizontalalignment='right', x=1.0)
#xticks(np.array([ep.end[0]-1e6, ep.end[0]]), ['0', str(int(ep.tot_length('s')))+' s'])
#ylabel("LMN", labelpad = -10)
#title("Wake", fontsize = 1)
xticks([])
gca().spines['bottom'].set_visible(False)

# ###################################################################
# # C REM
# ###################################################################
ex_rem = nts.IntervalSet(start = 15710150000, end= 15724363258)
ep = ex_rem

subplot(gs13[1,1])
noaxis(gca())

for k, n in enumerate(adn):
	spk = spikes[n].restrict(ex_rem).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)

tmp2 = decoding['rem'].rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
tmp2 = nts.Tsd(tmp2).restrict(ep)
plot(tmp2, '--', linewidth = 2, color = 'black', label = 'Decoded head-direction', alpha = alp)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
xticks([])
title("REM sleep")#, fontsize = 44)
gca().spines['bottom'].set_visible(False)
legend(frameon=False, handlelength = 1, bbox_to_anchor=(2,-0.05))

subplot(gs13[3,1])
noaxis(gca())

for k, n in enumerate(lmn):
	spk = spikes[n].restrict(ex_rem).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)

tmp2 = decoding['rem'].rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
tmp2 = nts.Tsd(tmp2).restrict(ep)
plot(tmp2, '--', linewidth = 2, color = 'black', alpha = alp)
plot(np.array([ep.end[0]-1e6, ep.end[0]]), [0, 0], linewidth = 3, color = 'black')
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
xticks([])
xlabel('1s', horizontalalignment='right', x=1.0)
gca().spines['bottom'].set_visible(False)


# ###################################################################
# # C SWS
# ###################################################################
#ex_sws = nts.IntervalSet(start = 4399905437.713542, end = 4403054216.186978)
ex_sws = nts.IntervalSet(start = 4400600000, end = 4403054216.186978)
ep = ex_sws
subplot(gs13[1,2])
simpleaxis(gca())

for k, n in enumerate(adn):
	spk = spikes[n].restrict(ep).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)
tmp2 = decoding['sws']
tmp3 = pd.Series(index = tmp2.index, data = np.unwrap(tmp2)).rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
tmp3 = tmp3%(2*np.pi)
tmp2 = nts.Tsd(tmp3).restrict(ep)
plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 2, color = 'black', alpha = alp)
plot(tmp2.loc[tmp2.idxmax()+30000:],'--', linewidth = 2, color = 'black', alpha = alp)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([])
#xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
xticks([])
title("non-REM sleep")#, fontsize = 44)
gca().spines['left'].set_visible(False)
gca().spines['bottom'].set_visible(False)


subplot(gs13[3,2])
simpleaxis(gca())

for k, n in enumerate(lmn):
	spk = spikes[n].restrict(ep).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = mks, markeredgewidth = 1.2)
tmp2 = decoding['sws']
tmp3 = pd.Series(index = tmp2.index, data = np.unwrap(tmp2)).rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
tmp3 = tmp3%(2*np.pi)
tmp2 = nts.Tsd(tmp3).restrict(ep)
plot(tmp2.loc[:tmp2.idxmax()],'--', linewidth = 2, color = 'black', alpha = alp)
plot(tmp2.loc[tmp2.idxmax()+30000:],'--', linewidth = 2, color = 'black', alpha = alp)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([])
#xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
xticks([])
gca().spines['left'].set_visible(False)
gca().spines['bottom'].set_visible(False)
plot(np.array([ep.end[0]-5e5, ep.end[0]]), [0, 0], linewidth = 3, color = 'black')
xlabel('0.5s', horizontalalignment='right', x=1.0)
# ylabel("HD neurons", labelpad = -10, fontsize = 10)
# title("non-REM sleep", fontsize = 1)


#################################################################################################################################
gs2 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outergs[0,1], wspace = 0.15)

################################
# Correaltion
gscor = gridspec.GridSpecFromSubplotSpec(3,2, subplot_spec = gs2[0,0], wspace = 0.2)

wakeremaxis = []
wakeswsaxis = []

paths = ['/home/guillaume/LMNphysio/data/All_correlation_ADN.pickle',
	'/home/guillaume/LMNphysio/data/All_correlation_ADN_LMN.pickle',
	'/home/guillaume/LMNphysio/data/All_correlation.pickle'
]
names = ['ADN', 'ADN/LMN', 'LMN']
clrs = ['lightgray', 'darkgray', 'gray']

for i, (p, n) in enumerate(zip(paths, names)):
	# 

	data3 = cPickle.load(open(p, 'rb'))
	allr = data3['allr']

	print(n, allr.shape)
	print(len(np.unique(np.array([[p[0].split('-')[0], p[1].split('-')[0]] for p in np.array(allr.index.values)]).flatten())))

	subplot(gscor[i,0])
	scatter(allr['wak'], allr['rem'], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0)
	m, b = np.polyfit(allr['wak'].values, allr['rem'].values, 1)
	x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
	r, p = scipy.stats.pearsonr(allr['wak'], allr['rem'])
	plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)))
	ylabel('REM corr. (r)')
	#title(n)
	text(-0.6, 0.5, n, horizontalalignment='center', verticalalignment='center', transform=gca().transAxes, fontsize = 21)	
	legend(handlelength = 0.3)
	ax = gca()
	aspectratio=1.0
	ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
	ax.set_aspect(ratio_default*aspectratio)
	locator_params(axis='y', nbins=3)
	locator_params(axis='x', nbins=3)
	if i == 2: xlabel('Wake corr. (r)')
	wakeremaxis.append(gca())
	# if i == 1:
	# 	text(0.5, 1.2, 'Pairwise correlation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

	# #
	subplot(gscor[i,1])
	scatter(allr['wak'], allr['sws'], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0)
	m, b = np.polyfit(allr['wak'].values, allr['sws'].values, 1)
	x = np.linspace(allr['wak'].min(), allr['wak'].max(),5)
	r, p = scipy.stats.pearsonr(allr['wak'], allr['sws'])
	plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)))
	if i == 2: xlabel('Wake corr. (r)')
	ylabel('non-REM corr. (r)')
	legend(handlelength = 0.1)
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



		# # Mean CC
		# subplot(gsswr[2,0])
		# cc_rip = cPickle.load(open('/home/guillaume/LMNphysio/data/LMN_SWR_CC.pickle', 'rb'))['allcc']
		# simpleaxis(gca())
		# gca().spines['left'].set_position('center')
		# m = cc_rip.mean(1).loc[-500:500]
		# s = cc_rip.std(1).loc[-500:500]

		# plot(m)
		# #fill_between(m.index.values, m-s, m+s, alpha = 0.3)
		# yticks([])
		# xlabel("Time from SWRs (ms)")

		# ####################
		# # Cross corr
		# gscor = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gs2[0,1], hspace = 0.5)

		# ####################
		# gsp2 = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec = gscor[1,0],width_ratios=[0.4, 0.05, 0.1,0.4])

		# # Population cross corr
		# subplot(gsp2[0,0])
		# simpleaxis(gca())
		# gca().spines['left'].set_position('center')

		# gcc = cPickle.load(open(os.path.join('../../data', 'All_GLOBAL_CC_ADN_LMN.pickle'), 'rb'))

		# tmp = gcc['cc_sws'].loc[-50:50]
		# tmp = (tmp - tmp.mean())/tmp.std()
		# plot(tmp, linewidth = 2, color = 'black', alpha = 0.3)
		# plot(tmp.mean(1), linewidth = 4, color = 'black', alpha = 1)
		# xlabel("LMN/ADN (ms)")
		# ylabel("z")
		# title("Population cc")
		# yticks([])
		# gca().text(0.0, 0.2, "non-REM sleep", transform = gca().transAxes, fontsize = 20, rotation = 90)	


		# acc = cPickle.load(open(os.path.join('../../data', 'All_crosscor_ADN_LMN.pickle'), 'rb'))


		# # Matrix cross corr
		# subplot(gsp2[0,2])
		# st = 'adn-lmn'
		# allpairs = acc['pairs']
		# subpairs = allpairs[allpairs['struct']==st]
		# group = subpairs.sort_values(by='ang diff').index.values
		# plot(allpairs.loc[group, 'ang diff'].values, np.arange(len(group))[::-1])
		# ylabel("Angular difference")
		# yticks([])
		# simpleaxis(gca())
		# xticks([0, np.pi], [0, 180])
		# subplot(gsp2[0,3])
		# allcc_sws = acc['cc_sws']
		# cc = allcc_sws[group]
		# cc = cc - cc.mean(0)
		# cc = cc / cc.std(0)
		# cc = cc.loc[-60:60]
		# tmp = scipy.ndimage.gaussian_filter(cc.T.values, (2, 2))
		# imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
		# title("Pairwise cc")
		# xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [int(cc.index[0]), 0, int(cc.index[-1])])
		# yticks([])
		# xlabel("LMN/ADN (ms)")

		# ##############################
		# # Exemples
		# ##############################
		# gsp = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = gscor[0,0])


		# allcc_wak = acc['cc_wak']
		# tcurves = acc['tcurves']

		# #idx = [1, 301]
		# idx = [1, 302]

		# for i in range(2):

		# 	axD = subplot(gsp[0,i])
		# 	# Cross correlogram SWS	
		# 	simpleaxis(gca())
		# 	gca().spines['left'].set_position('center')
		# 	cc = pd.concat([allcc_sws[group[idx[i]]].loc[-100:100],allcc_wak[group[idx[i]]].loc[-100:100]], 1)
		# 	cc = (cc - cc.mean())/cc.std()
		# 	#cc = cc.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)	
		# 	cc.iloc[:,1] = cc.iloc[:,1].rolling(window=30,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)	
		# 	plot(cc.iloc[:,0], linewidth = 2, color = 'red', label = 'non-REM')
		# 	plot(cc.iloc[:,1], linewidth = 2, color = 'black', label = 'Wake')
		# 	xlabel("LMN/ADN (ms)")
		# 	ylabel("z")

		# 	yticks([])
		# 	if i == 0: legend(frameon=False, handlelength = 0.5,bbox_to_anchor=(1.5,0.5))

		# 	cax1 = inset_axes(axD, "40%", "40%",					
		#                    bbox_to_anchor=(0.0, 0.6, 1, 1),
		#                    bbox_transform=axD.transAxes, 
		#                    loc = 'lower left',
		#                    axes_class = matplotlib.projections.get_projection_class('polar')
		#                    )
		# 	# Tuning curves	
		# 	tmp = tcurves[list(group[idx[i]])]
		# 	tmp = tmp/tmp.max()
		# 	cax1.plot(tmp.iloc[:,0], label = 'ADN', linewidth = 3)
		# 	cax1.plot(tmp.iloc[:,1], label = 'LMN', linewidth = 3)
		# 	if i == 0 : legend(frameon = False, handlelength = 0.4, bbox_to_anchor=(-0.05,1.1))
		# 	cax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
		# 	cax1.set_xticklabels([])
		# 	cax1.set_yticks([])


tax1.text(-0.2, 1.3, 'A', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21, weight="bold")		
tax1.text(3.6, 1.3, 'B', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21 , weight="bold")		
tax1.text(15.0, 1.3, 'E', horizontalalignment='center', verticalalignment='center', transform=tax1.transAxes, fontsize = 21, weight="bold")		

tax2.text(-0.2, 1.3, 'C', horizontalalignment='center', verticalalignment='center', transform=tax2.transAxes, fontsize = 21, weight="bold")		
tax2.text(3.6, 1.3, 'D', horizontalalignment='center', verticalalignment='center', transform=tax2.transAxes, fontsize = 21 , weight="bold")		



outergs.update(top= 0.94, bottom = 0.08, right = 0.98, left = 0.02)

savefig("/home/guillaume/Dropbox (Peyrache Lab)/Applications/Overleaf/Cosyne 2022 abstract submission/fig_1.pdf", dpi = 200, facecolor = 'white')
#show()
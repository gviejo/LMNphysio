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
	fig_height = fig_width*golden_mean*1.6          # height in inches
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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_KS25.txt'), delimiter = '\n', dtype = str, comments = '#')
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
	"axes.labelsize": 30,               # LaTeX default is 10pt font.
	"font.size": 20,
	"legend.fontsize": 15,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 15,
	"ytick.labelsize": 15,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 5,
	"axes.linewidth"        : 2,
	"ytick.major.size"      : 4,
	"xtick.major.size"      : 4
	}  
mpl.rcParams.update(pdf_with_latex)

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(2))

outergs = GridSpec(2,2, figure = fig)

####################################################################
# A WAKE
####################################################################
# ex_wake = nts.IntervalSet(start = 5.06792e+09, end = 5.10974e+09)
ex_wake = nts.IntervalSet(start = 7587976595.668784, end = 7604189853.273991)
ep = ex_wake
# gs1 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outergs[0,1])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)
subplot(outergs[0,0])
simpleaxis(gca())

for k, n in enumerate(adn):
	spk = spikes[n].restrict(ex_wake).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = 20, markeredgewidth = 3)
tmp = position['ry'].restrict(ep)
tmp	= tmp.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)	
plot(tmp, linewidth = 4, color = 'black')
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([0, 2*np.pi], ["0", "360"])
xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
ylabel("ADN", labelpad = -10)
title("Wake", fontsize = 44)


subplot(outergs[1,0])
simpleaxis(gca())

for k, n in enumerate(lmn):
	spk = spikes[n].restrict(ex_wake).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = 20, markeredgewidth = 3)
tmp = position['ry'].restrict(ep)
tmp	= tmp.rolling(window=40,win_type='gaussian',center=True,min_periods=1).mean(std=4.0)	
plot(tmp, linewidth = 4, color = 'black')
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([0, 2*np.pi], ["0", "360"])
xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
ylabel("LMN", labelpad = -10)
# title("Wake", fontsize = 1)



# ####################################################################
# # # B REM
# # ####################################################################
# ex_rem = nts.IntervalSet(start = 8.15664e+09, end = 8.19303e+09)
# ep = ex_rem
# subplot(outergs[0,1])
# simpleaxis(gca())

# for k, n in enumerate(adn):
# 	spk = spikes[n].restrict(ep).index.values
# 	if len(spk):
# 		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
# 		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, linewidth = 5, markeredgewidth = 1)

# ylim(0, 2*np.pi)
# xlim(ep.loc[0,'start'], ep.loc[0,'end'])
# yticks([0, 2*np.pi], ["0", "360"])
# xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])

# title("REM sleep", fontsize = 12)


# subplot(outergs[1,1])
# simpleaxis(gca())

# for k, n in enumerate(lmn):
# 	spk = spikes[n].restrict(ep).index.values
# 	if len(spk):
# 		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
# 		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, linewidth = 5, markeredgewidth = 1)
# ylim(0, 2*np.pi)
# xlim(ep.loc[0,'start'], ep.loc[0,'end'])
# yticks([0, 2*np.pi], ["0", "360"])
# xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])


# ###################################################################
# # C SWS
# ###################################################################
ex_sws = nts.IntervalSet(start = 4399305437.713542, end = 4403054216.186978)
ep = ex_sws
subplot(outergs[0,1])
simpleaxis(gca())

for k, n in enumerate(adn):
	spk = spikes[n].restrict(ep).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = 20, markeredgewidth = 3)

ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([0, 2*np.pi], ["0", "360"])
xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])

title("non-REM sleep", fontsize = 44)


subplot(outergs[1,1])
simpleaxis(gca())

for k, n in enumerate(lmn):
	spk = spikes[n].restrict(ep).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, markersize = 20, markeredgewidth = 3)
plot(position['ry'].restrict(ep), linewidth = 2, color = 'black')
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
yticks([0, 2*np.pi], ["0", "360"])
xticks(list(ep.loc[0].values), ['0', str(int(ep.tot_length('s')))+' s'])
# ylabel("HD neurons", labelpad = -10, fontsize = 10)
# title("non-REM sleep", fontsize = 1)




outergs.update(top= 0.94, bottom = 0.06, right = 0.98, left = 0.06)

savefig("/home/guillaume/Dropbox (Peyrache Lab)/Talks/2021-II/fig_ipn_final.png", dpi = 200, facecolor = 'white')
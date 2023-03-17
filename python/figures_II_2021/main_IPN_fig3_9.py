#!/usr/bin/env python


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


data_directory 	= '/mnt/DataRAID/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

session = 'Mouse32/Mouse32-140822'

lfp_hpc 		= pd.read_hdf(data_directory+session+'/'+session.split("/")[1]+'_EEG_SWR.h5')

data = cPickle.load(open('/home/guillaume/ThalamusPhysio/figures/figures_articles_v4/figure1/good_100ms_pickle/'+session.split("/")[1]+'.pickle', 'rb'))

iwak		= data['swr'][0]['iwak']
iswr		= data['swr'][0]['iswr']
rip_tsd		= data['swr'][0]['rip_tsd']
rip_spikes	= data['swr'][0]['rip_spikes']
times 		= data['swr'][0]['times']
wakangle	= data['swr'][0]['wakangle']
neurons		= data['swr'][0]['neurons']
tcurves		= data['swr'][0]['tcurves']
irand 		= data['rnd'][0]['irand']
iwak2 		= data['rnd'][0]['iwak2']

# sys.exit()

tcurves = tcurves.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)

peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		

# rip_tsd = pd.Serrip_tsd.index.values)

colors = np.hstack((np.linspace(0, 1, int(len(times)/2)), np.ones(1), np.linspace(0, 1, int(len(times)/2))[::-1]))

colors = np.arange(len(times))

H = wakangle.values/(2*np.pi)

HSV = np.vstack((H*360, np.ones_like(H)*85, np.ones_like(H)*45)).T

# from matplotlib.colors import hsv_to_rgb

# RGB = hsv_to_rgb(HSV)
RGB = np.array([hsluv.hsluv_to_rgb(HSV[i]) for i in range(len(HSV))])

# 4644.8144
# 4924.4720
# 5244.9392
# 7222.9480
# 7780.2968
# 11110.1888
# 11292.3240
# 11874.5688

good_ex = (np.array([4644.8144,4924.4720,5244.9392,7222.9480,7780.2968,11110.1888,11292.3240,11874.5688])*1e6).astype('int')

exemple = [np.where(i == rip_tsd.index.values)[0][0] for i in [good_ex[0],good_ex[1],good_ex[2]]]


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
	"axes.labelsize": 14,               # LaTeX default is 10pt font.
	"font.size": 14,
	"legend.fontsize": 14,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 14,
	"ytick.labelsize": 14,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preffontamble
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

fig = figure(figsize = figsize(1))

outergs = GridSpec(1,1, figure = fig)#, width_ratios = [0.3,0.7])#, height_ratios = [0.6, 0.3, 0.3], hspace = 0.4)#, width_ratios = [0.25, 0.75], wspace = 0.3)

# gs_top = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[0,0], width_ratios = [0.3, 0.7])#, height_ratios = [0.2, 0.8], hspace = 0)

####################################################################
# A TUNING CURVES
####################################################################
# gs_left = gridspec.GridSpecFromSubplotSpec(6,4, subplot_spec = outergs[0,0], hspace = 0.1, wspace = 0.1)

# for i, n in enumerate(neurons[::-1]):
# 	subplot(gs_left[int(i/4),i%4], projection = 'polar')
# 	clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
# 	gca().grid(zorder=0)
# 	xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
# 	yticks([])
# 	fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n].index.values), tcurves[n].values, color = clr, alpha = 0.8, linewidth =0, zorder=2)	
# 	# gca().text(0.75, 1, str(n), transform = gca().transAxes, fontsize = 10)

####################################################################
# B WAKE
####################################################################
subplot(outergs[0,0])
simpleaxis(gca())
session = 'Mouse32/Mouse32-140822'
generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
shankStructure 	= loadShankStructure(generalinfo)
if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
else:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
spikes,shank	= loadSpikeData(data_directory+session, shankStructure['thalamus'])		
n_channel,fs, shank_to_channel = loadXML(data_directory+session)	
wake_ep 		= loadEpoch(data_directory+session, 'wake')
sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])
position 		= pd.read_csv(data_directory+session+"/"+session.split("/")[1] + ".csv", delimiter = ',', header = None, index_col = [0])
angle 			= nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')

# tmp 			= (angle.values * len(neurons))/(2*np.pi)
# angle 			= nts.Tsd(t = angle.index.values, d = tmp)

spikes 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
# neurons 		= np.sort(list(spikes.keys()))
# neurons = tcurves.keys()

ep = nts.IntervalSet(start = 3.69296e+9, end = 3.69606e+9)
for k, n in enumerate(neurons):	
	spk = spikes[n].restrict(ep).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, linewidth = 15, markeredgewidth = 3)
# plot(angle.restrict(ep), linewidth = 2, color = 'black')
# xlim(times[0], times[-1])
# ylim(-1, len(neurons)+1)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
# xticks([-500,0,500], fontsize = 6)
# xlabel("Time from SWR (ms)")
yticks([0, 2*np.pi], ["0", "360"], fontsize = 16)
xticks(list(ep.loc[0].values), ['0', str(ep.tot_length('s'))+' s'])
# if i == 0:
ylabel("HD neurons", labelpad = -10, fontsize = 16)
title("non-REM sleep", fontsize = 16)





outergs.update(top= 0.9, bottom = 0.08, right = 0.95, left = 0.1)

savefig("/home/guillaume/Dropbox (Peyrache Lab)/Talks/2020-IPN/fig_ipn_3_9.png", dpi = 200, facecolor = 'white')

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

def smoothAngle(tsd, sd):
	tmp 			= pd.Series(index = tsd.index.values, data = np.unwrap(tsd.values))	
	tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=sd)
	newtsd			= nts.Tsd(tmp2%(2*np.pi))
	return newtsd


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


############################################################################################## 
# GENERAL infos
###############################################################################################
data_directory = '/mnt/DataGuillaume/'
datasets = np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

s = 'ADN-POSTSUB/A0028/A0028-140313A'



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
# sws_ep								= loadEpoch(path, 'sws')
# rem_ep								= loadEpoch(path, 'rem')


tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)

tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

tokeep, stat = findHDCells(tuning_curves, z = 20, p = 0.001 , m = 1)

neurons = np.intersect1d(np.where(shank>=8)[0], tokeep)


tcurves 							= tuning_curves[neurons]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]
neurons 							= [name+'_'+str(n) for n in tcurves.columns.values]
peaks.index							= pd.Index(neurons)
tcurves.columns						= pd.Index(neurons)






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
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 10,
	"ytick.labelsize": 7,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.5,
	"axes.linewidth"        : 1.2,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}  
mpl.rcParams.update(pdf_with_latex)

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(1.5))

outergs = GridSpec(2,1, figure = fig, height_ratios = [0.6, 0.4])#, width_ratios = [0.25, 0.75], wspace = 0.3)

gs_top = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[0,0], width_ratios = [0.25, 0.75])#, height_ratios = [0.2, 0.8], hspace = 0)
################################################################Yep ####
# A TUNING CURVES
####################################################################
gs_left = gridspec.GridSpecFromSubplotSpec(5,3, subplot_spec = gs_top[0,0])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)

for i, n in enumerate(neurons):
	subplot(gs_left[int(i/3),i%3], projection = 'polar')
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
# B SPIKES
####################################################################
subplot(gs_top[0,1])

angle = position['ry']

good_exs_wake = [nts.IntervalSet(start = [4.96148e+09], end = [4.99755e+09]),
				nts.IntervalSet(start = [3.96667e+09], end = [3.99714e+09]),
				nts.IntervalSet(start = [5.0872e+09], end = [5.13204e+09])
				]
ex_ep = good_exs_wake[0]


simpleaxis(gca())

newangle = smoothAngle(angle, 10)
plot(newangle.restrict(ex_ep), color = 'red', label = "Head-direction")
	
# SPIKES
for k, n in enumerate(neurons):
	spk = spikes[int(n.split("_")[1])]		
	clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])
	plot(spk.restrict(ex_ep).fillna(peaks[n]), '|', color = clr, markeredgewidth = 1, markersize = 3)
ylim(0, 2*np.pi)
yticks([0, np.pi, 2*np.pi], ['0', r"$\pi$", r"$2\pi$"])
xlim(ex_ep.loc[0].values)
# xt = np.arange(ex_ep[0].loc[0,'start'], ex_ep[0].loc[0,'end'], 15*1e6)
xtick = [0, 15, 30]
xlabel("Time (s)")





gs_bottom = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[1,0])#, width_ratios = [0.2, 0.8])#, height_ratios = [0.2, 0.8], hspace = 0)
####################################################################
# C CROSS-CORR SAME DIRECTION
####################################################################
subplot(gs_bottom[0,0])

####################################################################
# B SPIKES
####################################################################
subplot(gs_bottom[0,1])




outergs.update(top= 0.97, bottom = 0.07, right = 0.95, left = 0.05)

savefig("../../figures/figures_ipn_2020/fig_ipn_2.pdf", dpi = 200, facecolor = 'white')
savefig("/home/guillaume/Dropbox (Peyrache Lab)/Talks/2020-IPN/fig_ipn_2.png", dpi = 200, facecolor = 'white')
# os.system("evince ../../figures/figures_poster_2019/fig_poster_1.pdf &")


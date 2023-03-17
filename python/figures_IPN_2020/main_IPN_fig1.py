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
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*3          # height in inches
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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
infos_lmn = getAllInfos(data_directory, datasets)

datasets = np.loadtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
infos_adn = getAllInfos(data_directory, datasets)

datasets = np.loadtxt(os.path.join(data_directory,'datasets_POSTSUB.list'), delimiter = '\n', dtype = str, comments = '#')
infos_pos = getAllInfos(data_directory, datasets)


s_adn = 'LMN-ADN/A5001/A5001-200210C'

s_lmn = 'LMN/A1411/A1411-200908A'

s_pos = 'ADN-POSTSUB/A0028/A0028-140313A'

all_tcurves = []
all_peaks = []

for s, infos in zip([s_lmn, s_adn, s_pos],[infos_lmn,infos_adn, infos_pos]):
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
	# sleep_ep 							= loadEpoch(path, 'sleep')					
	# sws_ep								= loadEpoch(path, 'sws')
	# rem_ep								= loadEpoch(path, 'rem')

	tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
	# for i in tuning_curves:
	tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

	peaks = pd.Series(index=tuning_curves.columns,data = np.array([circmean(tuning_curves.index.values, tuning_curves[i].values) for i in tuning_curves.columns])).sort_values()		

	all_tcurves.append(tuning_curves)
	all_peaks.append(peaks)

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

fig = figure(figsize = figsize(0.75))

outergs = GridSpec(3,1, figure = fig, hspace = 0.5)#, width_ratios = [0.25, 0.75], wspace = 0.3)

# LMN
peaks = all_peaks[0]
tcurves = all_tcurves[0]
n = 1
subplot(outergs[2,0], projection = 'polar')
clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
gca().grid(zorder=0)

yticks([])
fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n].index.values), tcurves[n].values, color = "orangered", alpha = 0.8, linewidth =0, zorder=2)	
# xticks([0, 2*np.pi], ["0", r"$2\pi$"], fontsize = 6)
xticks([0, np.pi/2, np.pi, 3*np.pi/2], ["0", "90", "180", "270"], fontsize = 12)
# yticks([],[])


gca().text(0.75, 1, str(int(tcurves[n].max()))+" Hz", transform = gca().transAxes, fontsize = 12)#, fontweight='bold')


# ADN
peaks = all_peaks[1]
tcurves = all_tcurves[1]
n = 4
subplot(outergs[1,0], projection = 'polar')
clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
gca().grid(zorder=0)

yticks([])
fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n].index.values), tcurves[n].values, color = 'palevioletred', alpha = 0.8, linewidth =0, zorder=2)	
xticks([0, np.pi/2, np.pi, 3*np.pi/2], ["0", "90", "180", "270"], fontsize = 10)


gca().text(0.75, 1, str(int(tcurves[n].max()))+" Hz", transform = gca().transAxes, fontsize = 12)#, fontweight='bold')
	

# POS
peaks = all_peaks[2]
tcurves = all_tcurves[2]
n = 14
subplot(outergs[0,0], projection = 'polar')
clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])	
gca().grid(zorder=0)

yticks([])
fill_between(tcurves[n].index.values, np.zeros_like(tcurves[n].index.values), tcurves[n].values, color = 'forestgreen', alpha = 0.8, linewidth =0, zorder=2)	
# xticks([0, 2*np.pi], ["0", r"$2\pi$"], fontsize = 6)
xticks([0, np.pi/2, np.pi, 3*np.pi/2], ["0", "90", "180", "270"], fontsize = 10)
# yticks([],[])


gca().text(0.75, 1, str(int(tcurves[n].max()))+" Hz", transform = gca().transAxes, fontsize = 12)#, fontweight='bold')



outergs.update(top= 0.97, bottom = 0.07, right = 0.97, left = 0.02)

savefig("../../figures/figures_ipn_2020/fig_ipn_1.pdf", dpi = 200, facecolor = 'white')
savefig("/home/guillaume/Dropbox (Peyrache Lab)/Talks/2020-IPN/fig_ipn_1.png", dpi = 200, facecolor = 'white')
# os.system("evince ../../figures/figures_poster_2019/fig_poster_1.pdf &")


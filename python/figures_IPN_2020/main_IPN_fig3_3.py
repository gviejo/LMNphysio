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
	fig_height = fig_width*golden_mean*0.4          # height in inches
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
datasets = np.loadtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#')
infos = getAllInfos(data_directory, datasets)

# s = 'LMN/A1411/A1411-200908A'
s = 'LMN-ADN/A5002/A5002-200304A'

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


############################################################################################### 
# COMPUTING TUNING CURVES
###############################################################################################
tuning_curves, velocity, bins_velocity = computeLMNAngularTuningCurves(spikes, position['ry'], wake_ep, 121)
for i in tuning_curves:
	tuning_curves[i] = smoothAngularTuningCurves(tuning_curves[i], 20, 4)


tokeep, stat = findHDCells(tuning_curves[1], z = 10, p = 0.001 , m = 1)

tokeep = np.intersect1d(np.where(shank==3)[0], tokeep)

neurons = tokeep

shank = shank.flatten()

tcurves 							= tuning_curves[1][tokeep]
peaks 								= pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		
tcurves 							= tcurves[peaks.index.values]
# neurons 							= [name+'_'+str(n) for n in tcurves.columns.values]



data = cPickle.load(open('../../figures/figures_ipn_2020/fig_1_decoding.pickle', 'rb'))

data2 = cPickle.load(open(os.path.join('../../figures/figures_ipn_2020', 'All_crosscor.pickle'), 'rb'))

# tcurves		 		= data2['tcurves']
pairs 				= data2['pairs']
sess_groups	 		= data2['sess_groups']
# frates		 		= data2['frates']
cc_wak		 		= data2['cc_wak']
cc_rem		 		= data2['cc_rem']
cc_sws		 		= data2['cc_sws']
# peaks 				= data2['peaks']



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
	"axes.labelsize": 10,               # LaTeX default is 10pt font.
	"font.size": 9,
	"legend.fontsize": 9,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 9,
	"ytick.labelsize": 9,
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

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(2))

outergs = GridSpec(1,4, figure = fig, width_ratios = [0.1, 0.75, 0.75, 0.75], wspace = 0.2)

# angular differences
gs0 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outergs[0,0])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)
subplot(gs0[1,0])
simpleaxis(gca())
plot(pairs.values, np.arange(len(pairs))[::-1])
xticks([0, np.pi], ['0', r'180'])
yticks([0, len(pairs)-1], [len(pairs), 1])
xlabel("Ang. diff.", labelpad = -0.5)
ylabel("Pairs", labelpad = -20)
ylim(0, len(pairs)-1)


####################################################################
# # A WAKE
# ####################################################################
ex_wake = nts.IntervalSet(start = 5.06792e+09, end = 5.10974e+09)
ep = ex_wake
gs1 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outergs[0,1])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)
subplot(gs1[0,0])
simpleaxis(gca())

for k, n in enumerate(neurons):
	spk = spikes[n].restrict(ex_wake).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, linewidth = 5, markeredgewidth = 1)
plot(position['ry'].restrict(ep), linewidth = 2, color = 'black')
# xlim(times[0], times[-1])
# ylim(-1, len(neurons)+1)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
# xticks([-500,0,500], fontsize = 6)
# xlabel("Time from SWR (ms)")
yticks([0, 2*np.pi], ["0", "360"], fontsize = 10)
xticks(list(ep.loc[0].values), ['0', str(ep.tot_length('s'))+' s'])
# if i == 0:
ylabel("HD neurons", labelpad = -10, fontsize = 10)
title("Wake", fontsize = 12)



# for i, epoch, cc in zip(range(3), ['WAKE', 'REM', 'NREM'], [cc_wak, cc_rem, cc_sws]):
subplot(gs1[1,0])
simpleaxis(gca())
tmp = cc_wak[pairs.index]
# tmp = cc_wak
tmp = tmp - tmp.mean(0)
tmp = tmp / tmp.std(0)
tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))

imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
times = cc_wak.index.values
xticks([0, np.where(times==0)[0], len(times)], ['-10', '0', '10'])	
yticks([0, len(pairs)-1], [1, len(pairs)])
# title(epoch)
xlabel("Time lag (ms)")


####################################################################
# # B REM
# ####################################################################
ex_rem = nts.IntervalSet(start = 8.15664e+09, end = 8.19303e+09)
ep = ex_rem
gs2 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outergs[0,2])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)
subplot(gs2[0,0])

simpleaxis(gca())

for k, n in enumerate(neurons):
	spk = spikes[n].restrict(ex_rem).index.values
	if len(spk):
		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, linewidth = 5, markeredgewidth = 1)

# plot(position['ry'].restrict(ep), linewidth = 2, color = 'black')
plot(data['rem'].restrict(ep), linewidth = 2, color = 'red', alpha = 0.5)

# xlim(times[0], times[-1])
# ylim(-1, len(neurons)+1)
ylim(0, 2*np.pi)
xlim(ep.loc[0,'start'], ep.loc[0,'end'])
# xticks([-500,0,500], fontsize = 6)
# xlabel("Time from SWR (ms)")
yticks([0, 2*np.pi], ["0", "360"], fontsize = 10)
xticks(list(ep.loc[0].values), ['0', str(ep.tot_length('s'))+' s'])
# if i == 0:
# ylabel("HD neurons", labelpad = -10, fontsize = 10)
title("REM sleep", fontsize = 12)


# subplot(gs2[1,0])
# simpleaxis(gca())
# tmp = cc_rem[pairs.index]
# # tmp = cc_wak
# tmp = tmp - tmp.mean(0)
# tmp = tmp / tmp.std(0)
# tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))

# imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
# times = cc_rem.index.values
# xticks([0, np.where(times==0)[0], len(times)], ['-10', '0', '10'])	
# yticks([0, len(pairs)-1], [1, len(pairs)])
# # title(epoch)
# xlabel("Time lag (ms)")


# ####################################################################
# # # C SWS
# ####################################################################
# ex_sws = nts.IntervalSet(start = 6.58541e+09, end = 6.58785e+09)
# ep = ex_sws
# gs3 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outergs[0,3])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)
# subplot(gs3[0,0])

# simpleaxis(gca())

# for k, n in enumerate(neurons):
# 	spk = spikes[n].restrict(ex_sws).index.values
# 	if len(spk):
# 		clr = hsluv.hsluv_to_rgb([tcurves[n].idxmax()*180/np.pi,85,45])
# 		plot(spk, np.ones_like(spk)*tcurves[n].idxmax(), '|', color = clr, linewidth = 5, markeredgewidth = 1)

# # plot(position['ry'].restrict(ep), linewidth = 2, color = 'black')
# # plot(data['sws'].restrict(ep), linewidth = 2, color = 'red', alpha = 0.5)

# # xlim(times[0], times[-1])
# # ylim(-1, len(neurons)+1)
# ylim(0, 2*np.pi)
# xlim(ep.loc[0,'start'], ep.loc[0,'end'])
# # xticks([-500,0,500], fontsize = 6)
# # xlabel("Time from SWR (ms)")
# yticks([0, 2*np.pi], ["0", "360"])
# xticks(list(ep.loc[0].values), ['0', str(ep.tot_length('s'))+' s'])
# # if i == 0:
# # ylabel("HD neurons", labelpad = -10, fontsize = 10)
# title("non-REM sleep", fontsize = 12)


# subplot(gs3[1,0])
# simpleaxis(gca())
# tmp = cc_sws[pairs.index].loc[-1500:1500]
# times = tmp.index.values
# tmp = tmp - tmp.mean(0)
# tmp = tmp / tmp.std(0)
# tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))

# imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
# xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])])	
# yticks([0, len(pairs)-1], [1, len(pairs)])
# # title(epoch)
# xlabel("Time lag (ms)")

# sys.exit()


# tmp2 = cc_sws[pairs.index].loc[-1500:1500].copy()
# times = tmp2.index.values
# tmp2 = tmp2.values
# tmp3 = np.vstack((tmp2[times<-40],tmp2[times>40]))

# tmp3 = tmp3 - tmp3.mean(0)
# tmp3 = tmp3 / tmp3.std(0)
# tmp3 = scipy.ndimage.gaussian_filter(tmp3, (4, 4))


# tmp3 = np.array_split(tmp3, 2)
# tmp4 = np.vstack((tmp3[0], np.ones((1,tmp2.shape[1]))*np.nan, tmp3[1]))




# imshow(tmp4.T, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
# # times = cc_sws.index.values[xindex]
# xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])], fontsize = 6)
# yticks([0, len(pairs)-1], [1, len(pairs)], fontsize = 6)
# # title(epoch)
# xlabel("Time lag (ms)", fontsize = 7)



outergs.update(top= 0.92, bottom = 0.12, right = 0.96, left = 0.03)

savefig("/home/guillaume/Dropbox (Peyrache Lab)/Talks/2020-IPN/fig_ipn_3_3.png", dpi = 200, facecolor = 'white')
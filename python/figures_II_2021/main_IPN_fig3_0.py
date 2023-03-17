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
	golden_mean = (np.sqrt(5.0)-1.0)/3            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1          # height in inches
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

# sys.exit()

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
	"xtick.labelsize": 7,
	"ytick.labelsize": 7,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 1,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}  
mpl.rcParams.update(pdf_with_latex)

markers = ['d', 'o', 'v']

fig = figure(figsize = figsize(1.0))

# outergs = GridSpec(2,1, figure = fig, wspace = 0.3)

####################################################################
# A TUNING CURVES ADN
####################################################################
# gs_left = gridspec.GridSpecFromSubplotSpec(2,4, subplot_spec = outergs[0,0])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)

adn = peaks[adn].sort_values().index.values

for i, n in enumerate(adn):
	# subplot(gs_left[int(i/4),i%4], projection = 'polar')
	subplot(4,8,i+1, projection = 'polar')
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
	

	# gca().text(0.75, 1, str(int(tcurves[n].max()))+" Hz", transform = gca().transAxes, fontsize = 9)#, fontweight='bold')

####################################################################
# B TUNING CURVES LMN
####################################################################
# gs_left = gridspec.GridSpecFromSubplotSpec(4,4, subplot_spec = outergs[1,0])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)

lmn = peaks[lmn].sort_values().index.values

for i, n in enumerate(lmn):
	# subplot(gs_left[int(i/4),i%4], projection = 'polar')
	subplot(4,8, i+len(adn)+8+1, projection = 'polar')
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
	

	# gca().text(0.75, 1, str(int(tcurves[n].max()))+" Hz", transform = gca().transAxes, fontsize = 9)#, fontweight='bold')

	
# sys.exit()

####################################################################
# # B DECODING
# ####################################################################
# gs_right = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = outergs[0,1], hspace = 0.25)#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)


# for i, angle2, epoch, ex_ep in zip(range(3), [angle_wak, angle_rem, angle_sws], ['WAKE', 'REM', 'NREM'], [good_exs_wake, good_exs_rem, good_exs_sws]):

# 	if epoch is 'WAKE':
# 		gs2 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = gs_right[i,:])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)	
# 	elif epoch in ['REM', 'NREM']:
# 		gs2 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = gs_right[i,:], height_ratios = [0.2, 0.1, 0.8])#, width_ratios = [0.1, 0.5, 0.5, 0.5], height_ratios = [0.2, 0.8], hspace = 0)	


# 	if epoch is 'WAKE':
# 		subplot(gs2[0,:])
# 	elif epoch in ['REM', 'NREM']:
# 		subplot(gs2[2,:])

# 	simpleaxis(gca())
# 	# gca().spines['bottom'].set_visible(False)
# 	newangle = smoothAngle(angle, 10)
	
# 	if epoch is 'NREM':
# 		newangle2 = smoothAngle(angle2, 10)
# 	else:
# 		newangle2 = smoothAngle(angle2, 4)

# 	plot(newangle.restrict(ex_ep[0]), color = 'red', label = "Actual HD")
	
# 	if epoch == 'NREM':
# 		count = spike_counts.restrict(ex_ep[0]).sum(1)
# 		tmp = newangle2.restrict(ex_ep[0])
# 		idx = count.index.values[np.where(count < 0.3)[0]]
# 		tmp.loc[idx] = np.nan
# 		plot(tmp, color = 'gray')
# 		# sys.exit()
# 	else:
# 		plot(newangle2.restrict(ex_ep[0]), color = 'gray', label = "Decoded HD")


# 	if i == 0:
# 		legend(frameon = False)


# 	# SPIKES
# 	for k, n in enumerate(neurons):
# 		spk = spikes[n]		
# 		clr = hsluv.hsluv_to_rgb([peaks[n]*180/np.pi,85,45])
# 		plot(spk.restrict(ex_ep[0]).fillna(peaks[n]), '|', color = clr, markeredgewidth = 1, markersize = 3)
# 	ylim(0, 2*np.pi)
# 	yticks([0, np.pi, 2*np.pi], ['0', r"$\pi$", r"$2\pi$"])
# 	ylabel(epoch, rotation = 0, weight="bold", labelpad = 10, y = 0.75)
# 	xlim(ex_ep[0].loc[0].values)
# 	if epoch == 'WAKE':
# 		xt = np.arange(ex_ep[0].loc[0,'start'], ex_ep[0].loc[0,'end'], 15*1e6)
# 		xtick = [0, 15, 30]
# 	elif epoch == 'REM':
# 		xt = np.arange(ex_ep[0].loc[0,'start'], ex_ep[0].loc[0,'end'], 5*1e6)
# 		xtick = [0, 5, 10]
# 	elif epoch == 'NREM':
# 		xt = np.arange(ex_ep[0].loc[0,'start'], ex_ep[0].loc[0,'end'], 1*1e6)
# 		xtick = [0, 1, 2]		
# 		xlabel("Time (s)")

# 	xticks(xt, xtick)#, np.arange(len(xt)))

# 	# sys.exit()

# 	if epoch in ['REM', 'NREM']:

# 		subplot(gs2[0,:])
# 		noaxis(gca())
# 		plot(lfp.restrict(ex_ep[0]), color = 'black', linewidth = 0.6)
# 		xlim(ex_ep[0].loc[0].values)

# 		subplot(gs2[1,:])
# 		simpleaxis(gca())

# 		plot(ratio.restrict(ex_ep[0]), color = 'grey', linewidth = 1)
# 		# ylim(-1.1, 1.1)		
# 		xlim(ex_ep[0].loc[0].values)
# 		ylabel(r"$log(\frac{\theta}{\delta})$", rotation = 0, labelpad = 12, y = 0)

# 		axhline(0, color = 'black', linewidth = 0.78)
# 		yticks([0], [0])

# 		gca().spines['bottom'].set_visible(False)
# 		gca().set_xticks([])		


# 		# tmp = lfp.restrict(new_ep).values
# 		# tmp = lfp.restrict(ex_ep[0])

# 		# wp = pywt.WaveletPacket(tmp, 'db2', 'symmetric', maxlevel = 4)
# 		# f, t, Sxx = spectrogram_lspopt(tmp, 250, c_parameter=10)
		
# 		# wv = wigner_ville_spectrum(tmp, 1/250, frequency_divider = 6, smoothing_filter = 'gauss');imshow(np.sqrt(abs(wv)), aspect = 'auto', cmap = 'magma');show()
# 		# plot(tmp)
# 		# noaxis(gca())
# 		# pcolormesh(t, f, Sxx)
# 		# mtspec(tmp, 1/250, 3);	

# 		# sh1ow()
# 		# sys.exit()
# 		# spectrum, freqs, t, im = specgram(tmp, NFFT=64, noverlap=32, Fs = 250)
# 		# ylim(0, np.max(np.where(freqs>30)[0]))
# 		# ylim(0, 30)
# 		# noaxis(gca())

# outergs.update(top= 0.98, bottom = 0.07, right = 0.97, left = 0.02)

tight_layout()

savefig("/home/guillaume/Dropbox (Peyrache Lab)/Talks/2021-II/fig_ipn_3_0.png", dpi = 200, facecolor = 'white')
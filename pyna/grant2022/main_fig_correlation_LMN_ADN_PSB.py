# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-08-09 14:33:28
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-08-11 16:02:27
import numpy as np
import pandas as pd
import pynapple as nap
import sys
sys.path.append("../")
from functions import *

from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from itertools import combinations
from scipy.stats import zscore




data_dir = [
    '/mnt/Data2/', '/mnt/DataGuillaume', 
    '/mnt/DataGuillaume', '/mnt/DataGuillaume'
    ]

dataset_list = [
    'datasets_LMN_PSB.list', 'datasets_ADN.list',
    'datasets_LMN.list', 'datasets_LMN_ADN.list',
    ]


allr = {s:[] for s in ['lmn', 'adn', 'psb']}

for i in range(len(dataset_list)):

    ############################################################################################### 
    # GENERAL infos
    ###############################################################################################
    datasets = np.genfromtxt(os.path.join('/mnt/DataGuillaume',dataset_list[i]), delimiter = '\n', dtype = str, comments = '#')
        
    for s in datasets:
        print(s)
        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        path = os.path.join(data_dir[i], s)
        data = nap.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')
        try: 
            up_ep = data.read_neuroscope_intervals('up')
            down_ep = data.read_neuroscope_intervals('down')    
        except:
            up_ep, down_ep = None, None

        groups = data.spikes._metadata.groupby("location").groups

        grp = np.intersect1d(['adn', 'lmn', 'psb'], list(groups.keys()))

        for st in grp:
            idx = data.spikes._metadata[data.spikes._metadata["location"].str.contains(st)].index.values
            spikes = data.spikes[idx]
            
            ############################################################################################### 
            # COMPUTING TUNING CURVES
            ###############################################################################################
            tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
            tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
            
            SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position['ry'].time_support.loc[[0]], minmax=(0,2*np.pi))
            spikes.set_info(SI)
            r = correlate_TC_half_epochs(spikes, position['ry'], 120, (0, 2*np.pi))
            spikes.set_info(halfr = r)

            spikes = spikes.getby_threshold('SI', 0.15).getby_threshold('halfr', 0.7)

            tokeep = spikes.keys()
            if len(tokeep)>=2:

                groups = spikes._metadata.loc[tokeep].groupby("location").groups
                
                tcurves         = tuning_curves[tokeep]
                peaks           = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[j].values) for j in tcurves.columns]))

                try:
                    velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
                    newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
                except:
                    velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
                    newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)
                
                
                ############################################################################################### 
                # PEARSON CORRELATION
                ###############################################################################################
                rates = {}
                for e, ep, bin_size, std in zip(['wak', 'rem', 'sws'], [newwake_ep, rem_ep, sws_ep], [0.1, 0.1, 0.02], [2, 2, 5]):
                    count = spikes.count(bin_size, ep)
                    rate = count/bin_size        
                    #rate = zscore_rate(rate)
                    rate = rate.apply(zscore)
                    rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
                    rates[e] = nap.TsdFrame(rate, time_support = ep)

                if up_ep is not None and down_ep is not None:
                    rates['up'] = rates['sws'].restrict(up_ep)
                    rates['down'] = rates['sws'].restrict(down_ep)

                # idx=np.sort(np.random.choice(len(rates["sws"]), len(rates["down"]), replace=False))    
                # rates['rnd'] = rates['sws'].iloc[idx,:]

                
                # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
                pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))    
                pairs = pd.MultiIndex.from_tuples(pairs)
                r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)
                for p in r.index:
                    for ep in rates.keys():
                        r.loc[p, ep] = scipy.stats.pearsonr(rates[ep][int(p[0])],rates[ep][int(p[1])])[0]

                name = data.basename
                pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
                pairs = pd.MultiIndex.from_tuples(pairs)
                r.index = pairs
                            
                #######################
                # SAVING
                #######################
                allr[st].append(r)

for k in allr.keys():
    allr[k] = pd.concat(allr[k], 0)


datatosave = {'allr':allr}
cPickle.dump(datatosave, open(os.path.join('../../data/', 'All_correlation_ADN_LMN_PSB.pickle'), 'wb'))


# ################################################################################################
# # FIGURES
# ################################################################################################
# from matplotlib.pyplot import *
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import matplotlib.font_manager as font_manager
# #matplotlib.style.use('seaborn-paper')

# def figsize(scale):
#     fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
#     inches_per_pt = 1.0/72.27                       # Convert pt to inch
#     golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
#     #fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
#     fig_width = 7
#     fig_height = fig_width*golden_mean*1.2         # height in inches
#     fig_size = [fig_width,fig_height]
#     return fig_size

# def simpleaxis(ax):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     # ax.xaxis.set_tick_params(size=6)
#     # ax.yaxis.set_tick_params(size=6)

# def noaxis(ax):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # ax.xaxis.set_tick_params(size=6)
#     # ax.yaxis.set_tick_params(size=6)

# font_dir = ['/home/guillaume/Dropbox/CosyneData/figures_poster_2022']
# for font in font_manager.findSystemFonts(font_dir):
#     font_manager.fontManager.addfont(font)

# fontsize = 7

# rcParams['font.family'] = 'Helvetica'
# rcParams['font.size'] = fontsize
# rcParams['axes.labelsize'] = fontsize
# rcParams['axes.labelpad'] = 3
# #rcParams['axes.labelweight'] = 'bold'
# rcParams['axes.titlesize'] = fontsize
# rcParams['xtick.labelsize'] = fontsize
# rcParams['ytick.labelsize'] = fontsize
# rcParams['legend.fontsize'] = fontsize
# rcParams['figure.titlesize'] = fontsize
# rcParams['xtick.major.size'] = 1.3
# rcParams['ytick.major.size'] = 1.3
# rcParams['xtick.major.width'] = 0.4
# rcParams['ytick.major.width'] = 0.4
# rcParams['axes.linewidth'] = 0.6
# rcParams['axes.edgecolor'] = 'grey'
# rcParams['axes.axisbelow'] = True
# # rcParams['xtick.color'] = 'grey'
# # rcParams['ytick.color'] = 'grey'


# allaxis=[]

# clrs = ['steelblue', 'gray', 'darkorange']

# mkrs = 6

# eps = ['REM corr. (r)', 'NREM corr. (r)']

# fig = figure(figsize = figsize(2))
# gs = GridSpec(3,4)
# for i,st in enumerate(['lmn', 'adn', 'psb']):
#     for j,ep in enumerate(['rem', 'sws']):
#         subplot(gs[i,j])
#         simpleaxis(gca())
#         scatter(allr[st]['wak'], allr[st][ep], color = clrs[i], alpha = 0.5, edgecolor = None, linewidths=0, s = mkrs)
#         m, b = np.polyfit(allr[st]['wak'].values, allr[st][ep].values, 1)
#         x = np.linspace(allr[st]['wak'].min(), allr[st]['wak'].max(),5)
#         r, p = scipy.stats.pearsonr(allr[st]['wak'], allr[st][ep])
#         plot(x, x*m + b, color = 'red', label = 'r = '+str(np.round(r, 2)), linewidth = 1)
#         xlabel('Wake corr. (r)')
#         ylabel(eps[j])
#         legend(handlelength = 0.4, loc='center', bbox_to_anchor=(0.2, 0.7, 0.5, 0.5), framealpha =0)
#         ax = gca()
#         aspectratio=1.0
#         ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
#         #ax.set_aspect(ratio_default*aspectratio)
#         ax.set_aspect(1)
#         locator_params(axis='y', nbins=3)
#         locator_params(axis='x', nbins=3)
#         #xlabel('Wake corr. (r)')
#         allaxis.append(gca())
            
# xlims = []
# ylims = []
# for ax in allaxis:
#     xlims.append(ax.get_xlim())
#     ylims.append(ax.get_ylim())
# xlims = np.array(xlims)
# ylims = np.array(ylims)
# xl = (np.min(xlims[:,0]), np.max(xlims[:,1]))
# yl = (np.min(ylims[:,0]), np.max(ylims[:,1]))
# for ax in allaxis:
#     ax.set_xlim(xl)
#     ax.set_ylim(yl)
# # outergs.update(top= 0.97, bottom = 0.06, right = 0.96, left = 0.06)

# tight_layout()
# gs.update(hspace=0.4)

# savefig("/home/guillaume/LMNphysio/figures/figures_adrien_2022/fig1.pdf", dpi = 200, facecolor = 'white')
# #show()



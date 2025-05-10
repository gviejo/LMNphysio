# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2025-01-04 06:11:33
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-05-06 11:22:59
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Apply the default theme
sns.set_context("notebook")

############################################################################################### 
# GENERAL infos
###############################################################################################
if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'
elif os.path.exists('/Users/gviejo/Data'):
    data_directory = '/Users/gviejo/Data'    

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#')


allcc = {'wak':[], 'all':[], 'sws':[]}
allccdown = {'psb':[], 'lmn':[]}

angdiff = {}

hd_info = {}

alltcurves = []
alltcahv = []

for s in datasets:

    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    # filepath = os.path.join(path, "pynapplenwb", basename + ".nwb")

    if os.path.exists(filepath):
        
        nwb = nap.load_file(filepath)
        
        spikes = nwb['units']
        spikes = spikes.getby_threshold("rate", 1)

        position = []
        columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        for k in columns:
            position.append(nwb[k].values)
        position = np.array(position)
        position = np.transpose(position)
        position = nap.TsdFrame(
            t=nwb['x'].t,
            d=position,
            columns=columns,
            time_support=nwb['position_time_support'])

        epochs = nwb['epochs']
        wake_ep = epochs[epochs.tags == "wake"]
        sws_ep = nwb['sws']
        rem_ep = nwb['rem']    

        nwb.close()

        spikes = spikes[(spikes.location=="psb")|(spikes.location=="lmn")]
        
        
        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
        spikes.set_info(SI)

        # spikes = spikes[spikes.SI>0.1]


        # CHECKING HALF EPOCHS
        wake2_ep = splitWake(position.time_support.loc[[0]])    
        tokeep2 = []
        stats2 = []
        tcurves2 = []   
        for i in range(2):
            tcurves_half = nap.compute_1d_tuning_curves(
                spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
                ep = wake2_ep[i]
                )
            tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 4)

            tokeep, stat = findHDCells(tcurves_half)
            tokeep2.append(tokeep)
            stats2.append(stat)
            tcurves2.append(tcurves_half)       
        tokeep = np.intersect1d(tokeep2[0], tokeep2[1])  
        
        spikes = spikes[tokeep]

        psb = spikes.index[spikes.location=="psb"]
        lmn = spikes.index[spikes.location=="lmn"]
        
        tcurves = tuning_curves[tokeep]
        # tcurves = tuning_curves



        # Filtering by SI only for LMN
        # lmn = np.intersect1d(lmn[spikes.SI[lmn]>0.1], tokeep)


        try:
            velocity = computeLinearVelocity(position[['x', 'z']], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.003).time_support.drop_short_intervals(1)
        except:
            velocity = computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2)
            newwake_ep = velocity.threshold(0.07).time_support.drop_short_intervals(1)


        ############################################################################################### 
        # CROSS CORRELOGRAM
        ###############################################################################################         
        
        
        for e, ep, bin_size, window_size in zip(['wak', 'sws', 'all'], 
            [newwake_ep, sws_ep, spikes.time_support], 
            [0.0005, 0.0005, 0.0005], [0.1, 0.1, 0.1]):

            tmp = nap.compute_crosscorrelogram(
                    # tuple(spikes.getby_category("location").values()),
                    (spikes[psb], spikes[lmn]),
                    bin_size, 
                    window_size,
                    ep, norm=False)


            pairs = [(basename + "_" + str(n), basename + "_" + str(m)) for n, m in tmp.columns]
            pairs = pd.MultiIndex.from_tuples(pairs)
            tmp.columns = pairs
            
            allcc[e].append(tmp)

            # if e == 'sws':
            #     sys.exit()
            #     cc1 = nap.compute_crosscorrelogram(spikes[[1, lmn[0]]], 0.0001, 1, spikes.time_support, norm=True)
            #     cc2 = nap.compute_crosscorrelogram(spikes[[1, lmn[0]]], 0.001, 1, spikes.time_support, norm=True)
            #     cc3 = nap.compute_crosscorrelogram(spikes[[1, lmn[0]]], 0.01, 1, spikes.time_support, norm=True)


        #######################
        # Angular differences
        #######################
        peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        for p in pairs:
            i = int(p[0].split("_")[1])
            j = int(p[1].split("_")[1])
            angdiff[p] = min(np.abs(peaks[i] - peaks[j]), 2*np.pi-np.abs(peaks[i] - peaks[j]))
            
            hd_info[p] = SI.loc[[i,j]].values.flatten()


        tcurves.columns = [basename + "_" + str(n) for n in tcurves.columns]
        alltcurves.append(tcurves)

        ######################
        # AHV
        ######################
        angle = position['ry']
        ahv = np.gradient(np.unwrap(angle).bin_average(0.05))/np.mean(np.diff(angle.t))

        tcahv = nap.compute_1d_tuning_curves(spikes, ahv, 100, wake_ep, minmax=(-50, 50))

        tcahv.columns = [basename + "_" + str(n) for n in tcahv.columns]
        alltcahv.append(tcahv)


        # sys.exit()


for e in allcc.keys():
    allcc[e] = pd.concat(allcc[e], axis=1)

angdiff = pd.Series(angdiff)
hd_info = pd.DataFrame(hd_info).T
hd_info.columns = ['lmn', 'psb']

alltcurves = pd.concat(alltcurves, axis=1)
alltcahv = pd.concat(alltcahv, axis=1)



# Detecting synaptic connections
def get_zscore(cc):
    # Smoothing with big gaussian
    sigma = int(0.01/np.median(np.diff(cc.index.values)))+1
    df = cc.apply(lambda col: gaussian_filter1d(col, sigma=sigma))

    # Zscoring
    tmp = cc - df
    zcc = tmp/tmp.std(0)
    return zcc


# Detecting
thr = 3.0
zcc = get_zscore(allcc['sws'])
tmp2 = (zcc>=thr).loc[0.002:0.008]
pc = tmp2.columns[np.any(tmp2 & tmp2.shift(1, fill_value=False), 0)]
zorder = zcc[pc].loc[0.002:0.008].max().sort_values()
order = zorder[::-1].index.values

# Counter 
tmp3 = (zcc>=thr).loc[-0.008:-0.002]
pc = tmp3.columns[np.any(tmp3 & tmp3.shift(1, fill_value=False), 0)]
zcounter = zcc[pc].loc[-0.008:-0.002].max().sort_values()
counter = zcounter[::-1].index.values

# cc = allcc['sws']
# # Smoothing with big gaussian
# sigma = int(0.01/np.median(np.diff(cc.index.values)))
# df = cc.apply(lambda col: gaussian_filter1d(col, sigma=sigma))
# # Getting upper bound from df
# upper_bounds = pd.DataFrame(data=scipy.stats.poisson.ppf(0.95, df)-1,index=df.index,columns=df.columns)
# upper_bounds[upper_bounds<0] = 0
# tmp = (cc>upper_bounds).loc[0.002:0.008]
# pc = tmp.columns[np.any(tmp & tmp.shift(1, fill_value=False), 0)]



# figure()
# for i, p in enumerate(pc):
#     subplot(5,7,i+1)
#     plot(cc[p].loc[-0.03:0.03])
#     plot(df[p].loc[-0.03:0.03])
#     plot(upper_bounds[p].loc[-0.03:0.03])

# figure()
# # p = ('A3018-220614B_44', 'A3018-220614B_49')
# p = ('A3019-220630A_14', 'A3019-220630A_85')

# plot(cc[p].loc[-0.06:0.06], '.-')
# plot(df[p].loc[-0.06:0.06])
# plot(upper_bounds[p].loc[-0.06:0.06])

pdf_filename = os.path.expanduser("~/Dropbox/LMNphysio/summary_cc/fig_synaptic_detection.pdf")

with PdfPages(pdf_filename) as pdf:

    figure(figsize=(15,10))
    for i, p in enumerate(order[0:35]):
        subplot(5,7,i+1)
        plot(zcc[p].loc[-0.03:0.03], color='blue')
        axhline(thr)
    tight_layout()
    pdf.savefig(gcf())
    close(gcf())



    figure(figsize=(15,10))
    for i, p in enumerate(counter[0:35]):
        subplot(5,7,i+1)
        plot(zcc[p].loc[-0.03:0.03], color='green')
        axhline(thr)
    tight_layout()        
    pdf.savefig(gcf())
    close(gcf())


    figure()
    hist(zorder, bins = np.linspace(2.9, np.max(zorder), 20), label="+lag", color='blue')
    hist(zcounter, bins = np.linspace(2.9, np.max(zorder), 20), label="-lag", color='green')
    legend()
    xlabel("zscore")
    tight_layout()
    pdf.savefig(gcf())
    close(gcf())

    figure()
    subplot(121)
    hist(zcc[order].loc[0.002:0.008].idxmax(), label="+lag", color='blue')
    xlim(-0.01, 0.01)
    xlabel("time lag")
    ylim(0, 15)
    legend()
    subplot(122)
    hist(zcc[counter].loc[-0.008:-0.002].idxmax(), label="-lag", color='green')
    xlim(-0.01, 0.01)
    ylim(0, 15)
    legend()
    xlabel("time lag")
    tight_layout()
    pdf.savefig(gcf())
    close(gcf())


    figure()    
    maxs = zcc[order].loc[0.002:0.008].max()
    maxw = get_zscore(allcc['wak'])[order].loc[0.002:0.008].max()
    hist(maxs, label='sleep')
    hist(maxw,  label='wake')
    legend()
    ylabel("zcore)")
    tight_layout()
    pdf.savefig(gcf())
    close(gcf())

    figure()
    a = zcc[order].loc[0.002:0.008].idxmax().sort_values().index
    imshow(zcc[a].loc[-0.05:0.05].values.T, vmax=3, cmap='jet')
    tight_layout()
    pdf.savefig(gcf())
    close(gcf())


# sys.exit()

datatosave = {
    'allcc':allcc, 
    'angdiff':angdiff, 
    'alltc':alltcurves, 
    "pospeak": zcc[order].loc[0.002:0.008].idxmax(), 
    "negpeak":zcc[counter].loc[-0.008:-0.002].idxmax(),
    "order":order,
    "counter":counter,
    "zcc":zcc
    }

dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'CC_LMN-PSB.pickle'), 'wb'))


sys.exit()

pdf_filename = os.path.expanduser("~/Dropbox/LMNphysio/summary_cc/fig_CC_LMN_PSB.pdf")

with PdfPages(pdf_filename) as pdf:
    sns.set(font_scale=0.3)


    fig = figure()
    for i, e in enumerate(allcc.keys()):
        subplot(1,3,i+1)
        #plot(allcc[e], alpha = 0.7, color = 'grey')
        plot(allcc[e].mean(1), '.-')
        title(e)
    
    pdf.savefig(fig)
    close(fig)



    # figure()
    # for i,k in enumerate(['psb', 'lmn']):
    #     subplot(2,1,i+1)
    #     plot(allccdown[k].mean(1))
    # show()


    angbins = np.linspace(0, np.pi, 4)
    idx = np.digitize(angdiff, angbins)-1

    ang0 = angdiff[idx==0].index

    cc = allcc['sws']#[ang0]

    cc = cc[cc.idxmax().sort_values().index] 

    # imshow(cc.values.T, aspect='auto')

    # bins = np.linspace(hd_info['psb'].min(), hd_info['psb'].max(), 9)
    # bins = np.geomspace(hd_info['psb'].min(), hd_info['psb'].max(), 9)

    # idx = np.digitize(hd_info['psb'], bins)-1

    # ccg = {}
    # for i in np.unique(idx):
    #     ccg[i] = allcc['sws'][hd_info.index[idx==i]].mean(1)
    # ccg = pd.DataFrame(ccg)



    # figure()
    # for i in range(ccg.shape[1]):
    #     subplot(3,3,i+1)
    #     plot(ccg[i].loc[-0.1:0.1])


    cc = allcc['sws']
    maxt = cc.idxmax()

    idx = maxt[(maxt<0.015) & (maxt>0.0)].index.values

    new_idx = cc[idx].max().sort_values().index.values[::-1]

    # new_idx = allcc['all'][idx]maxt[maxt.sort_values()>0].sort_values().index.values


    figure()
    for i in range(80):
        ax = subplot(8,10,i+1)    
        p = new_idx[i]
        plot((allcc['sws'][p]-df[p]).loc[-0.02:0.02], linewidth=0.4)
        axvline(0, linewidth=0.2)
        title(p[0].split("_")[0] + " " + str(i))
        xticks([])
        yticks([])
    tight_layout()
    pdf.savefig(gcf())
    close(gcf())


    figure()
    for i in range(60):
        ax = subplot(6,10,i+1, projection='polar')    
        p = new_idx[i]
        plot(alltcurves[p[0]])
        plot(alltcurves[p[1]])
        title(p[0].split("_")[0] + " " + str(i))        
        xticks([])
        yticks([])

    tight_layout()
    pdf.savefig(gcf())
    close(gcf())

    figure()
    for i in range(54):
        ax = subplot(6,9,i+1)
        p = new_idx[i]
        plot(alltcahv[p[0]].loc[-40:40], label='lmn')
        plot(alltcahv[p[1]].loc[-40:40], label='psb')
        title(i)
        xticks([])
        yticks([])

    tight_layout()
    pdf.savefig(gcf())
    close(gcf())

    # for p in [0, 2]:

    #     figure(figsize=(18,10))
    #     subplot(121, projection='polar')
    #     plot(alltcurves[new_idx[p][0]], label='lmn')
    #     plot(alltcurves[new_idx[p][1]], label='psb')
    #     legend()
    #     # subplot(132)
    #     # plot(alltcahv[new_idx[p][0]].loc[-40:40], label='lmn')
    #     # plot(alltcahv[new_idx[p][1]].loc[-40:40], label='psb')
    #     # legend()        
    #     subplot(122)
    #     plot(allcc['sws'][new_idx[p]].loc[-0.02:0.02], linewidth=0.5)
    #     plot(allcc['wak'][new_idx[p]].loc[-0.02:0.02], linewidth=0.5)
    #     grid()

    #     tight_layout()
    #     pdf.savefig(gcf())
    #     close(gcf())



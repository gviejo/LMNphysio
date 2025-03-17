# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-29 13:46:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-03-17 18:23:27
# %%
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from pylab import *
import sys, os
sys.path.append("..")
from functions import *
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import yaml
import warnings
warnings.simplefilter('ignore', category=UserWarning)



# %%
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

datasets = yaml.safe_load(
    open(os.path.join(
        data_directory,
        "datasets_OPTO.yaml"), "r"))['opto']


SI_thr = {
    'adn':0.2, 
    'lmn':0.1,
    'psb':1.0
    }

allr = {}
corr = {}
allfr = {}
alltc = {}
allsi = {}

# allfr = {"wake":[], "sleep":[]}
# allmeta = {"wake":[], "sleep":[]}
# alltc = {"wake":[], "sleep":[]}

# %%
for st in ['adn', 'lmn']:

    allr[st] = {}
    corr[st] = {}
    allfr[st] = {}
    alltc[st] = {}
    allsi[st] = {}

    for gr in ['opto', 'ctrl']:    

        allr[st][gr] = {}
        corr[st][gr] = {}
        allfr[st][gr] ={}
        alltc[st][gr] = {}
        allsi[st][gr] = {}

        for sd in ['ipsi', 'bilateral']:

            try:
                dataset = datasets[st][gr][sd]['wake']
            except:
                dataset = []

            if len(dataset):


                allr[st][gr][sd] = []
                corr[st][gr][sd] = []
                allfr[st][gr][sd] = []
                alltc[st][gr][sd] = []
                alltc[st][gr][sd] = {'opto':[], 'pre':[]}
                allsi[st][gr][sd] = [] 

                for s in dataset:
                    print(s)
                    ############################################################################################### 
                    # LOADING DATA
                    ###############################################################################################
                    path = os.path.join(data_directory, "OPTO", s)
                    basename = os.path.basename(path)
                    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
                    
                    if not os.path.exists(filepath):
                        print("missing ", st, gr, s)
                        continue
                    else:

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
                        opto_ep = nwb["opto"]
                        
                        
                            
                    spikes = spikes[spikes.location == st]
                    stim_duration = 10.0
                    opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start'])>=stim_duration-0.001]
                    

                    ############################################################################################### 
                    # COMPUTING TUNING CURVES
                    ###############################################################################################
                    # Opto ep is always the second wake
                    assert len(opto_ep.intersect(position.time_support[0])) == 0

                    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
                    tuning_curves = smoothAngularTuningCurves(tuning_curves)
                    tcurves = tuning_curves
                    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
                    spikes.set_info(SI)
                    spikes.set_info(max_fr = tcurves.max())
                    
                    # figure()
                    # for i in range(len(spikes)):
                    #     subplot(6,6,i+1, projection='polar')
                    #     plot(tuning_curves[spikes.keys()[i]])
                    #     title(np.round(spikes.SI[spikes.keys()[i]], 3))
                    # show()

                    spikes = spikes.getby_threshold("SI", SI_thr[st])
                    spikes = spikes.getby_threshold("rate", 1.0)
                    spikes = spikes.getby_threshold("max_fr", 3.0)

                    tokeep = spikes.index
                    tcurves = tcurves[tokeep]
                    tuning_curves = tcurves
                    peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
                    order = np.argsort(peaks.values)
                    spikes.set_info(order=order, peaks=peaks)
                    tuning_curves.columns = [basename+"_"+str(i) for i in tuning_curves.columns]
                    
                    ############################################################################################### 
                    # COMPUTING OPTO TUNING CURVES
                    ###############################################################################################
                    opto_tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = opto_ep)
                    opto_tuning_curves = smoothAngularTuningCurves(opto_tuning_curves)            
                    SI_opto = nap.compute_1d_mutual_info(opto_tuning_curves, position['ry'], opto_ep, (0, 2*np.pi))

                    opto_tuning_curves.columns = [basename+"_"+str(i) for i in opto_tuning_curves.columns]
                    
                    SI = pd.concat([spikes.SI, SI_opto], axis=1)
                    SI.columns = ['pre', 'opto']
                    SI.index = tuning_curves.columns                    
                    
                    ############################################################################################### 
                    # FIRING RATE MODULATION
                    ###############################################################################################    

                    # peth = nap.compute_perievent(spikes[tokeep], nap.Ts(opto_ep.start), minmax=(-stim_duration, 2*stim_duration))
                    # frates = pd.DataFrame({n:np.sum(peth[n].count(0.05), 1).values for n in peth.keys()})

                    frates = nap.compute_eventcorrelogram(spikes, nap.Ts(opto_ep.start), stim_duration/20., stim_duration*2, norm=True)

                    frates.columns = [basename+"_"+str(i) for i in frates.columns]

                    ############################################################################################### 
                    # PEARSON CORRELATION
                    ###############################################################################################        
                    if len(tokeep) > 3:
                        
                        velocity = np.abs(computeAngularVelocity(position['ry'], position.time_support.loc[[0]], 0.2))
                        newwake_ep = velocity.threshold(0.02).time_support.drop_short_intervals(1).merge_close_intervals(1)

                        pre_ep = nap.IntervalSet(start=opto_ep.start-stim_duration, end=opto_ep.start)
                        # pre_ep = pre_ep.intersect(sws_ep)
                        pre_ep = pre_ep.drop_short_intervals(stim_duration-0.001)

                        rates = {}
                        for e, iset, bin_size, std in zip(['wak', 'pre', 'opto'], [newwake_ep, pre_ep, opto_ep], [0.3, 0.3, 0.3], [2, 2, 2]):
                            count = spikes.count(bin_size, iset)
                            rate = count/bin_size
                            rate = rate.smooth(std=bin_size*std).as_dataframe()
                            rate = rate.apply(zscore)                    
                            rates[e] = rate
                        
                        pairs = [basename+"_"+i+"-"+j for i,j in list(combinations(np.array(spikes.keys()).astype(str), 2))]
                        r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

                        for e in rates.keys():
                            tmp = np.corrcoef(rates[e].values.T)
                            if len(tmp):
                                r[e] = tmp[np.triu_indices(tmp.shape[0], 1)]

                        allr[st][gr][sd].append(r)

                        #######################
                        # Session correlation
                        #######################

                        tmp = pd.DataFrame(index=[basename])
                        tmp['pre'] = scipy.stats.pearsonr(r['wak'], r['pre'])[0]
                        tmp['opto'] = scipy.stats.pearsonr(r['wak'], r['opto'])[0]
                        tmp['n'] = len(spikes)
                        corr[st][gr][sd].append(tmp)

                    
                    #######################
                    # SAVING
                    #######################
                    allfr[st][gr][sd].append(frates)
                    alltc[st][gr][sd]['opto'].append(opto_tuning_curves)
                    alltc[st][gr][sd]['pre'].append(tuning_curves)
                    allsi[st][gr][sd].append(SI)
                    
                    # metadata = spikes._metadata
                    # metadata.index = frates.columns
                    # allmeta[ep].append(metadata)
                    # tuning_curves.columns = frates.columns
                    # alltc[ep].append(tuning_curves)       

                 
                allr[st][gr][sd] = pd.concat(allr[st][gr][sd], axis=0)
                corr[st][gr][sd] = pd.concat(corr[st][gr][sd], axis=0)
                allfr[st][gr][sd] = pd.concat(allfr[st][gr][sd], axis=1)
                allsi[st][gr][sd] = pd.concat(allsi[st][gr][sd], axis=0)

                alltc[st][gr][sd]['opto'] = pd.concat(alltc[st][gr][sd]['opto'], axis=1)
                alltc[st][gr][sd]['pre'] = pd.concat(alltc[st][gr][sd]['pre'], axis=1)
                                

                # allmeta[ep] = pd.concat(allmeta[ep], axis=0)
                # alltc[ep] = pd.concat(alltc[ep], axis=1)



#%%
change_fr = {}
for st in allfr.keys():
    change_fr[st] = {}    
    for gr in allfr[st].keys():
        change_fr[st][gr] = {}        
        for sd in allfr[st][gr]:
            change_fr[st][gr][sd] = pd.concat([
                allfr[st][gr][sd].loc[-0.8:0].mean(),
                allfr[st][gr][sd].loc[0:1].mean()], axis=1)
            change_fr[st][gr][sd].columns = ['pre', 'opto']
            
        

pdf_filename = os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_OPTO_WAKE.pdf")

with PdfPages(pdf_filename) as pdf:

    # Change firing rate over time
    #%%
    fig = figure()
    gs = GridSpec(2, 3)
    for i, keys in zip(range(3), [('adn', 'opto', 'ipsi'), ('adn', 'opto', 'bilateral'), ('lmn', 'opto', 'ipsi')]):
    # order = allmeta[ep].sort_values(by="SI").index.values
        # tmp = allfr[ep][order].loc[sl]   
        tmp = allfr[keys[0]][keys[1]][keys[2]].loc[-10:20]     
        tmp = tmp.apply(lambda x: gaussian_filter1d(x, sigma=1, mode='constant'))
        subplot(gs[0,i])
        plot(tmp, color = 'grey', alpha=0.2)
        plot(tmp.mean(1), color = 'blue')
        axvline(0)
        axvline(10)
        xlim(-10, 20)
        title("-".join(keys))
        subplot(gs[1,i])    
        # tmp = tmp - tmp.loc[msl].mean(0)
        # tmp = tmp / tmp.std(0)    
        imshow(tmp.values.T, cmap = 'jet', aspect='auto', vmin = 0, vmax = 2)
        colorbar()
        # title(ep)
    tight_layout()
    pdf.savefig(fig)
    close(fig)

    # TUning curves 1
    #%%
    fig = figure(figsize =(20, 15))
    gs = GridSpec(4, 3)
    for i, keys in zip(range(3), [('adn', 'opto', 'ipsi'), ('adn', 'opto', 'bilateral'), ('lmn', 'opto', 'ipsi')]):

        st, gr, sd = keys
            
        tc1 = centerTuningCurves_with_peak(alltc[st][gr][sd]['pre'])
        tc2 = centerTuningCurves_with_peak(alltc[st][gr][sd]['opto'], by=alltc[st][gr][sd]['pre'])
        order = tc1.loc[0].sort_values().index
        subplot(gs[0,i])
        imshow(tc1[order].values.T, aspect='auto')
        title("-".join(keys))
        if i == 0: ylabel("Ctrl")
        subplot(gs[1,i])
        imshow(tc2[order].values.T, aspect='auto')
        if i == 0: ylabel("Opto")
        subplot(gs[2,i])
        tc2 = tc2/tc1.max()
        tc1 = tc1/tc1.max()        
        for tc, lb in zip([tc2, tc1], ['opto', 'ctrl']):
            m = tc.mean(1)
            s = tc.std(1)
            plot(m, label=lb)
            fill_between(m.index.values, m-s, m+s, alpha=0.1)
        grid()
        legend()
        subplot(gs[3,i])
        widths = pd.concat([
            np.abs(np.abs(tc1.loc[:0]-0.5).idxmin()) + np.abs(np.abs(tc1.loc[0:]-0.5).idxmin()),
            np.abs(np.abs(tc2.loc[:0]-0.5).idxmin()) + np.abs(np.abs(tc2.loc[0:]-0.5).idxmin())
            ], axis=1)
        for n in widths.index.values:
            plot(np.arange(2), widths.loc[n].values, '-', color='grey', alpha=0.2)
            xticks([0,1], ['ctrl', 'opto'])
        plot(np.arange(2), widths.mean(0).values, '-', color='red')
        ylabel("Width (rad)")
        ylim(0, 2*np.pi)
        

    tight_layout()
    pdf.savefig(fig)
    close(fig)


    # TUning curves 2
    #%%
    fig = figure(figsize=(28, 12))
    gs = GridSpec(1, 3)
    for i, keys in zip(range(3), [('adn', 'opto', 'ipsi'), ('adn', 'opto', 'bilateral'), ('lmn', 'opto', 'ipsi')]):
        
        gs2 = GridSpecFromSubplotSpec(5, 4, gs[0,i])

        st, gr, sd = keys
            
        tc1 = alltc[st][gr][sd]['pre']
        tc2 = alltc[st][gr][sd]['opto']
        order = tc1.max().sort_values().index.values[0:20].reshape(5, 4)

        for j, k in np.ndindex(5,4):
            subplot(gs2[j,k])
            plot(tc2[order[j,k]], '--')
            plot(tc1[order[j,k]])

            if j == 0 and k == 0:
                title("-".join(keys))

    tight_layout()                
    pdf.savefig(fig)
    close(fig)

    # All correlations
    #%%
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


    fig = figure(figsize=(28, 12))
    gs = GridSpec(2, 1)
    gstop = GridSpecFromSubplotSpec(1, 3, gs[0,0])
    for i, keys in zip(range(3), [('lmn', 'opto', 'ipsi'), ('adn', 'opto', 'ipsi'), ('adn', 'opto', 'bilateral')]):
    # for i, st in enumerate(['adn', 'lmn']):
        gs2 = GridSpecFromSubplotSpec(1, 2, gstop[0,i])

        st, gr, sd = keys

        # Opto vs ctrl
        subplot(gs2[0,0])
        e = 'opto'
        for j, gr in enumerate(['opto', 'ctrl']):

            if sd not in allr[st][gr].keys():
                continue

            r, p = scipy.stats.pearsonr(allr[st][gr][sd]['wak'], allr[st][gr][sd][e]) # Wak vs opto

            plot(allr[st][gr][sd]['wak'], allr[st][gr][sd][e], 'o', color = cycle[j], alpha = 0.5, label = 'r = '+str(np.round(r, 3)) + f"; {gr}")

            m, b = np.polyfit(allr[st][gr][sd]['wak'].values, allr[st][gr][sd][e].values, 1)
            x = np.linspace(allr[st][gr][sd]['wak'].min(), allr[st][gr][sd]['wak'].max(),5)
            plot(x, x*m + b)
            xlabel('wak')
            ylabel(e)
            xlim(allr[st][gr][sd]['wak'].min(), allr[st][gr][sd]['wak'].max())
            # ylim(allr[st].iloc[:,1:].min().min(), allr[st].iloc[:,1:].max().max())
            ylim(allr[st][gr][sd]['wak'].min(), allr[st][gr][sd]['wak'].max())
            legend()
            title(f"{st.upper()} "+" chrimson vs tdtomato" + f"\n {sd}")
                
        # Opto vs preopto
        subplot(gs2[0,1])    
        gr = 'opto'
        labels = ['light', 'nolight']
        for k, e in enumerate(['opto', 'pre']):

            r, p = scipy.stats.pearsonr(allr[st][gr][sd]['wak'], allr[st][gr][sd][e])

            plot(allr[st][gr][sd]['wak'], allr[st][gr][sd][e], 'o', color = cycle[k], alpha = 0.5, label = 'r = '+str(np.round(r, 3)) + f"; {labels[k]}")

            m, b = np.polyfit(allr[st][gr][sd]['wak'].values, allr[st][gr][sd][e].values, 1)
            x = np.linspace(allr[st][gr][sd]['wak'].min(), allr[st][gr][sd]['wak'].max(),5)
            plot(x, x*m + b)
            xlabel('wak')
            ylabel(e)
            xlim(allr[st][gr][sd]['wak'].min(), allr[st][gr][sd]['wak'].max())
            # ylim(allr[st].iloc[:,1:].min().min(), allr[st].iloc[:,1:].max().max())
            ylim(allr[st][gr][sd]['wak'].min(), allr[st][gr][sd]['wak'].max())
            legend()
            title(f"{st.upper()} "+" chrimson light vs no light")

    gsbot = GridSpecFromSubplotSpec(1, 2, gs[1,0])
    orders = {
        "adn" : [('adn', 'opto', 'ipsi', 'opto'), ('adn', 'opto', 'ipsi', 'pre'), ('adn', 'ctrl', 'ipsi', 'opto'), ('adn', 'opto', 'bilateral', 'opto')],
        "lmn" : [('lmn', 'opto', 'ipsi', 'opto'), ('lmn', 'opto', 'ipsi', 'pre'), ('lmn', 'ctrl', 'ipsi', 'opto')]
    }
    for i, st in enumerate(['lmn', 'adn']):

        gs3 = GridSpecFromSubplotSpec(2, 2, gsbot[0,i])

        # Pearson per sesion
        subplot(gs3[:,0])
        for j, keys in enumerate(orders[st]):
            st, gr, sd, k = keys

            corr2 = corr[st][gr][sd]
            corr2 = corr2[corr2['n']>3][k]
        
            plot(np.random.randn(len(corr2))*0.1+np.ones(len(corr2))*j, corr2, 'o', markersize=6)

        ylim(-1, 1)
        # xticks(np.arange(len(orders[st])), ["\n".join(o) for o in orders[st]])
        if st == "adn":
            xticks([0, 1, 2, 3], ['chrimson\nipsilateral', 'chrimson\npre-opto', 'tdtomate\n(control)', 'chrimson\nbilateral'])
        else:
            xticks([0, 1, 2], ['chrimson\nipsilateral', 'chrimson\npre-opto', 'tdtomate\n(control)'])

        ylabel("Pearson r")
        title(st.upper())
        
        # # Firing rate
        subplot(gs3[0,1])
        count = 0    
        for j, keys in enumerate(orders[st]):
            st, gr, sd, k = keys
            if k == "pre": continue
            chfr = change_fr[st][gr][sd]
            delta = (chfr['opto'] - chfr['pre'])/chfr['pre']
            plot(np.random.randn(len(delta))*0.076+np.ones(len(chfr))*count, delta, 'o', markersize=6)
            count += 1        
        title(st.upper())
        ylabel("Delta firing rate")
        ylim(-1, 1)
        if st == "adn":
            xticks([0, 1, 2], ['chrimson\nipsilateral', 'tdtomate\n(control)', 'chrimson\nbilateral'])
        else:
            xticks([0, 1], ['chrimson\nipsilateral', 'tdtomate\n(control)'])
        
        # Change SI
        subplot(gs3[1,1])
        count = 0
        for j, keys in enumerate(orders[st]):
            st, gr, sd, k = keys
            if k == "pre": continue
            chsi = allsi[st][gr][sd]
            delta = (chsi['opto'] - chsi['pre'])/chsi['pre']
            plot(np.random.randn(len(delta))*0.076+np.ones(len(chsi))*count, delta, 'o', markersize=6)
            count += 1            
        ylabel("Delta SI")
        ylim(-1, 1)
        if st == "adn":
            xticks([0, 1, 2], ['chrimson\nipsilateral', 'tdtomate\n(control)', 'chrimson\nbilateral'])
        else:
            xticks([0, 1], ['chrimson\nipsilateral', 'tdtomate\n(control)'])



    gs.update(right=0.98, left=0.1)
    pdf.savefig(fig)
    close(fig)
    # savefig(os.path.expanduser("~/Dropbox/LMNphysio/summary_opto/fig_OPTO_WAKE.png"))
    # show()


# ##################################################################
# # FOR FIGURE 1
# ##################################################################
# datatosave = {
#     "allfr":allfr,
#     "allmeta":allmeta,
#     "alltc":alltc
# }


# dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
# file_name = "OPTO_PSB.pickle"

# import _pickle as cPickle

# with open(os.path.join(dropbox_path, file_name), "wb") as f:
#     cPickle.dump(datatosave, f)
# %%

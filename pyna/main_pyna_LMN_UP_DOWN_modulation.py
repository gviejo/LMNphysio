# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-03-01 12:03:19
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-05-13 17:23:05

# %%
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from pylab import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.stats import zscore



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

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    # np.genfromtxt(os.path.join(data_directory,'datasets_LMN_PSB.list'), delimiter = '\n', dtype = str, comments = '#'),
    ])


datasets = np.unique(datasets)



allr = []
pearson = pd.DataFrame(columns = ['sws', 'up', 'down', 'count', 'decimated'])
baseline = pd.DataFrame(columns = ['sws', 'up', 'down', 'decimated'])
frates = []



for s in datasets:
    print(s)
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)

    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")

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
        
        up_ep = read_neuroscope_intervals(path, basename, 'up')
        down_ep = read_neuroscope_intervals(path, basename, 'down')

        spikes = spikes[spikes.location == "lmn"]        


        ############################################################################################### 
        # COMPUTING TUNING CURVES
        ###############################################################################################
        tuning_curves = nap.compute_1d_tuning_curves(
            spikes, position['ry'], 120, minmax=(0, 2*np.pi), 
            ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)
        
        SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position.time_support.loc[[0]], minmax=(0,2*np.pi))
        spikes.set_info(SI)

        spikes = spikes[spikes.SI>0.1]

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
    
        if len(tokeep) > 4:

            spikes = spikes[tokeep]
            # groups = spikes._metadata.loc[tokeep].groupby("location").groups
            tcurves         = tuning_curves[tokeep]

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
            for e, ep, bin_size, std in zip(['wak', 'sws'], [newwake_ep, sws_ep], [0.2, 0.01], [2, 2]):
                ep = ep.drop_short_intervals(bin_size*21)
                count = spikes.count(bin_size, ep)
                rate = count/bin_size
                # rate = rate.as_dataframe()
                rate = rate.smooth(std=bin_size*std, windowsize=bin_size*20).as_dataframe()
                rate = rate.apply(zscore)            
                rates[e] = nap.TsdFrame(rate, time_support=ep)                
            
            # pairs = list(product(groups['adn'].astype(str), groups['lmn'].astype(str)))
            pairs = list(combinations(np.array(spikes.keys()).astype(str), 2))
            pairs = pd.MultiIndex.from_tuples(pairs)
            r = pd.DataFrame(index = pairs, columns = rates.keys(), dtype = np.float32)

            for ep in rates.keys():                
                # tmp = np.arctanh(np.corrcoef(rates[ep].values.T))
                tmp = np.corrcoef(rates[ep].values.T)
                r[ep] = tmp[np.triu_indices(tmp.shape[0], 1)]                            

            name = basename    
            pairs = list(combinations([name+'_'+str(n) for n in spikes.keys()], 2)) 
            pairs = pd.MultiIndex.from_tuples(pairs)
            r.index = pairs

            r_wak = r['wak']

            #######################
            # Angular differences
            #######################
            peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))                
            for p, (i, j) in zip(pairs, list(combinations(spikes.keys(), 2))):
                r.loc[p, 'ang'] = min(np.abs(peaks[i] - peaks[j]), 2*np.pi-np.abs(peaks[i] - peaks[j]))

            
            # #######################
            # # COMPUTING PEARSON R FOR EACH SESSION
            # #######################
            pearson.loc[s] = np.zeros((5))*np.nan
            pearson.loc[s, 'sws'] = scipy.stats.pearsonr(r['wak'], r['sws'])[0]

            ############################################################################################### 
            # PEARSON CORRELATION UP/DOWN 
            ###############################################################################################
            for e, ep in zip(['up', 'down'], [up_ep, down_ep]):
                # tmp = np.arctanh(np.corrcoef(rates[ep].values.T))
                rates[e] = rates['sws'].restrict(ep)
                tmp = np.corrcoef(rates[e].values.T)
                r[e] = tmp[np.triu_indices(tmp.shape[0], 1)] 



            pearson.loc[s, 'up'] = scipy.stats.pearsonr(r['wak'], r['up'])[0]
            pearson.loc[s, 'down'] = scipy.stats.pearsonr(r['wak'], r['down'])[0]
            pearson.loc[s, 'count'] = len(spikes)


            #########################
            # FIRING RATE MODULATION
            #########################
            delta = (spikes.restrict(down_ep).rate - spikes.restrict(up_ep).rate)/spikes.restrict(up_ep).rate


            ##############################
            # SPIKES DECIMATION
            ##############################
            
            up_spikes = spikes.restrict(up_ep)
            percent = 1+delta
            percent[percent>1] = 1.0
            bin_size = 0.01

            allr_dec = []
            r_dec = []

            for i in range(10):

                # Matching spike counts
                decimated_spikes = nap.TsGroup(
                    {n:nap.Ts(np.sort(np.random.choice(up_spikes[n].t, int(len(up_spikes[n])*percent[n]), replace=False))) for n in up_spikes
                    }, time_support=up_ep)
                        
                count = decimated_spikes.count(bin_size, up_ep.drop_short_intervals(bin_size*21))
                rate = count/bin_size
                rate = rate.smooth(std=bin_size*2, windowsize=bin_size*20).as_dataframe()
                rate = rate.apply(zscore)
                rate = nap.TsdFrame(rate, time_support = up_ep)            

                # Matching durations and number of ep
                # w = up_ep.end - up_ep.start
                # w = w/np.sum(w)
                # durations = down_ep.end - down_ep.start
                # for j in range(len(durations)):
                #     tmp = up_ep[np.random.choice(up_ep.index, size=1, p=w)[0]].values
                #     np.
                # a = nap.Ts(np.random.uniform(up_ep.start[0], up_ep.end[-1], 1000000)).restrict(up_ep).t
                # a = a[:len(a) - (len(a) % 2)]
                # a = nap.IntervalSet(a).drop_short_intervals(bin_size*21).intersect(up_ep)
                # a = a.merge_close_intervals(bin_size*22)
                
                new_up_ep = up_ep.split(np.mean(down_ep.end-down_ep.start))
                new_up_ep = new_up_ep[np.random.choice(np.arange(0, len(new_up_ep)), len(down_ep), replace=False)]

                # new_up_ep = nap.IntervalSet(down_ep.start-(down_ep.end-down_ep.start), down_ep.start).intersect(up_ep)
                rate = rate.restrict(new_up_ep)                        

                # Correlation                
                tmp = np.corrcoef(rate.values.T)
                r_ = tmp[np.triu_indices(tmp.shape[0], 1)]

                rd, p = scipy.stats.pearsonr(r_wak, r_)

                r_dec.append(r_)
                allr_dec.append(rd)

            r_dec = np.mean(r_dec, 0)
            pearson.loc[s,'decimated'] = np.mean(allr_dec)

            r["decimated"] = r_dec

            #######################
            # SHUFFLE BASELINE
            #######################
            for c in ['sws', 'up', 'down', 'decimated']:
                baseline.loc[s,c] = np.mean([scipy.stats.pearsonr(r['wak'].values, r[c].sample(frac=1, random_state=42).values)[0] for i in range(100)])


            #######################
            # SAVING
            #######################
            allr.append(r)
            delta.index = [name+"_"+str(n) for n in spikes.keys()]
            frates.append(delta)



allr = pd.concat(allr)

frates = pd.concat(frates)


datatosave = {    
    "pearson": pearson,
    "frates":frates,
    "baseline": baseline,
}
dropbox_path = os.path.expanduser("~/Dropbox/LMNphysio/data")
cPickle.dump(datatosave, open(os.path.join(dropbox_path, 'CORR_LMN-PSB_UP_DOWN.pickle'), 'wb'))




figure(figsize = (16, 10))
for i, e in enumerate(["sws", "up", "down"]):
    subplot(2, 4, i+1)
    tmp = allr[['wak', e]].dropna()
    plot(tmp['wak'], tmp[e], 'o', color = 'red', alpha = 0.5)
    m, b = np.polyfit(tmp['wak'].values, tmp[e].values, 1)
    x = np.linspace(tmp['wak'].min(), tmp['wak'].max(),5)
    plot(x, x*m + b)
    xlabel('wake')
    ylabel(e)
    xlim(-2.0, 2.0)    
    ylim(-2.0, 2.0)
    r, p = scipy.stats.pearsonr(tmp['wak'], tmp[e])
    title('r = '+str(np.round(r, 3))+f'\n p={p}')

subplot(2,4,4)
plot(np.zeros(len(pearson))+np.random.randn(len(pearson))*0.1, pearson['sws'].values, 'o')
plot(np.ones(len(pearson))+np.random.randn(len(pearson))*0.1, pearson['up'].values, 'o') 
plot(np.ones(len(pearson))*2+np.random.randn(len(pearson))*0.1, pearson['down'].values, 'o') 
plot(np.ones(len(pearson))*3+np.random.randn(len(pearson))*0.1, pearson['decimated'].values, 'o') 

plot([-0.1, 0.1], [pearson['sws'].mean(), pearson['sws'].mean()], linewidth=3)
plot([0.9, 1.1], [pearson['up'].mean(), pearson['up'].mean()], linewidth=3)
plot([1.9, 2.1], [pearson['down'].mean(), pearson['down'].mean()], linewidth=3)
plot([2.9, 3.1], [pearson['decimated'].mean()]*2, linewidth=3)

# print(scipy.stats.wilcoxon(pearson["rem"], pearson["sws"]))
# print(scipy.stats.ttest_ind(pearson["rem"], pearson["sws"]))

xticks([0, 1, 2, 3], ['sws', 'up', 'down', 'decimated'])
ylim(-1, 1)

subplot(2,4,5)
for i in pearson.index:
    plot([0, 1], pearson.loc[i, ['up','down']].values, 'o-')

for i in pearson.index:
    plot([2, 3], pearson.loc[i, ['decimated','down']].values, 'o-')


ylim(-1,1)
xticks([0, 1, 2, 3], ['up', 'down', 'up\ndecimated', 'down'])
xlim(-1, 4)

title(
    "p="+str(np.round(scipy.stats.wilcoxon(pearson['up'], pearson['down'], alternative='greater')[1],6))+" | "+
    "p="+str(np.round(scipy.stats.wilcoxon(pearson['decimated'], pearson['down'], alternative='greater')[1],6))
    )
ylabel("Pearson r")

subplot(2,4,6)
hist(frates*100, 20)
title("Modulation firing rate")

subplot(2,4,7)
for i in pearson.index:
    plot([0, 1], pearson.loc[i, ['up','down']].values-baseline.loc[i, ['up','down']].values, 'o-')

for i in pearson.index:
    plot([2, 3], pearson.loc[i, ['decimated','down']].values-baseline.loc[i, ['decimated','down']].values, 'o-')


ylim(-1,1)
xticks([0, 1, 2, 3], ['up', 'down', 'up\ndecimated', 'down'])
xlim(-1, 4)

tmp = (pearson[['decimated','down']]-baseline[['decimated','down']]).astype("float").values
title("p="+str(scipy.stats.wilcoxon(tmp[:,0], tmp[:,1], alternative="greater")[1]))

tight_layout()
# show()
savefig(
    os.path.expanduser("~/Dropbox/LMNphysio/summary_psb/fig_correlatio_up_down.pdf")
    )

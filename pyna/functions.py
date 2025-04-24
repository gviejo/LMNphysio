# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-02-28 16:16:36
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2025-04-22 17:48:35
import numpy as np
from numba import jit
import pandas as pd
import sys, os
import scipy
# from scipy import signal
from itertools import combinations
from pycircstat.descriptive import mean as circmean
from pylab import *
import pynapple as nap
from matplotlib.colors import hsv_to_rgb
# import xgboost as xgb
# from LinearDecoder import linearDecoder
from scipy.ndimage import gaussian_filter1d

def smooth_series(series, sigma=1):
    smoothed_series = gaussian_filter1d(series, sigma=sigma, mode='constant')
    return pd.Series(smoothed_series, index = series.index)

def getAllInfos(data_directory, datasets):
    allm = np.unique(["/".join(s.split("/")[0:2]) for s in datasets])
    infos = {}
    for m in allm:      
        path = os.path.join(data_directory, m)
        csv_file = list(filter(lambda x: '.csv' in x, os.listdir(path)))[0]
        infos[m.split('/')[1]] = pd.read_csv(os.path.join(path, csv_file), index_col = 0)
    return infos

def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
    new_tuning_curves = {}  
    for i in tuning_curves.columns:
        tcurves = tuning_curves[i]
        offset = np.mean(np.diff(tcurves.index.values))
        padded  = pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi)-offset,
                                                tcurves.index.values,
                                                tcurves.index.values+(2*np.pi)+offset)),
                            data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        # smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)
        smoothed = smooth_series(padded, sigma=deviation)
        new_tuning_curves[i] = smoothed.loc[tcurves.index]

    new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

    return new_tuning_curves

def splitWake(ep):
    if len(ep) != 1:
        print('Cant split wake in 2')
        sys.exit()
    tmp = np.zeros((2,2))
    tmp[0,0] = ep.values[0,0]
    tmp[1,1] = ep.values[0,1]
    tmp[1,0] = ep.values[0,0] + ep.tot_length()/2
    tmp[0,1] = tmp[1,0]
    tmp[0,1] -= 1e-3
    return nap.IntervalSet(start = tmp[:,0], end = tmp[:,1])

def zscore_rate(rate):
    time_support = rate.time_support
    idx = rate.index
    cols = rate.columns
    rate = rate.values
    rate = rate - rate.mean(0)
    rate = rate / rate.std(0)
    rate = nap.TsdFrame(t=idx.values, d=rate, time_support = time_support, columns = cols)
    return rate

def findHDCells(tuning_curves, z = 50, p = 0.0001 , m = 1):
    """
        Peak firing rate larger than 1
        and Rayleigh test p<0.001 & z > 100
    """
    cond1 = tuning_curves.max()>m
    from pycircstat.tests import rayleigh
    stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
    for k in tuning_curves:
        stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
    cond2 = np.logical_and(stat['pval']<p,stat['z']>z)
    tokeep = stat.index.values[np.where(np.logical_and(cond1, cond2))[0]]
    return tokeep, stat 

def computeLinearVelocity(pos, ep, bin_size):
    pos = pos.restrict(ep)
    pos2 = pos.bin_average(bin_size)    
    pos2 = pos2.smooth(1.0)
    speed = np.sqrt(np.sum(np.power(pos2.values[1:, :] - pos2.values[0:-1, :], 2), 1))    
    speed = nap.Tsd(t = pos2.index.values[0:-1], d=speed, time_support = ep)
    return speed

def computeAngularVelocity(angle, ep, bin_size):
    """this function only works for single epoch
    """        
    tmp = np.unwrap(angle.restrict(ep).values)
    tmp = pd.Series(index=angle.restrict(ep).index.values, data=tmp)
    tmp = smooth_series(tmp, 2.0)
    tmp = nap.Tsd(t = tmp.index.values, d = tmp.values)    
    tmp = tmp.bin_average(bin_size)
    t = tmp.index.values[0:-1]+np.diff(tmp.index.values)
    velocity = nap.Tsd(t=t, d = np.diff(tmp))
    return velocity


def getRGB(angle, ep, bin_size):
    tmp = np.unwrap(angle.values)
    uangle = nap.Tsd(t=angle.index.values, d=tmp, time_support=angle.time_support)
    uangle = uangle.bin_average(bin_size, ep)
    angle2 = nap.Tsd(t=uangle.index.values, d=uangle.values%(2*np.pi), time_support=uangle.time_support)
    H = angle2.values/(2*np.pi)
    HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    RGB = hsv_to_rgb(HSV)    
    return RGB

def getBinnedAngle(angle, ep, bin_size):
    angle = angle.restrict(ep)
    bins = np.arange(ep.as_units('s').start.iloc[0], ep.as_units('s').end.iloc[-1]+bin_size, bin_size)  
    tmp = angle.as_series().groupby(np.digitize(angle.as_units('s').index.values, bins)-1).mean()
    tmp2 = pd.Series(index = np.arange(0, len(bins)-1), dtype = float64)
    tmp2.loc[tmp.index.values] = tmp.values
    tmp2 = tmp2.fillna(method='ffill')
    tmp = nap.Tsd(t = bins[0:-1] + np.diff(bins)/2., d = tmp2.values, time_support = ep)
    return tmp

def xgb_decodage(Xr, Yr, Xt):      
    n_class = 32
    bins = np.linspace(0, 2*np.pi, n_class+1)
    labels = np.digitize(Yr.values, bins)-1
    dtrain = xgb.DMatrix(Xr.values, label=labels)
    dtest = xgb.DMatrix(Xt.values)

    params = {'objective': "multi:softprob",
    'eval_metric': "mlogloss", #loglikelihood loss
    'seed': 2925, #for reproducibility    
    'learning_rate': 0.01,
    'min_child_weight': 2, 
    # 'n_estimators': 1000,
    # 'subsample': 0.5,    
    'max_depth': 5, 
    'gamma': 0.5,
    'num_class':n_class}

    num_round = 50
    bst = xgb.train(params, dtrain, num_round)
    
    ymat = bst.predict(dtest)
    pclas = np.argmax(ymat, 1)

    clas = bins[0:-1] + np.diff(bins)/2

    Yp = clas[pclas]

    Yp = nap.Tsd(t = Xt.index.values, d = Yp, time_support = Xt.time_support)
    proba = nap.TsdFrame(t = Xt.index.values, d = ymat, time_support = Xt.time_support)

    return Yp, proba, bst

def xgb_predict(bst, Xt, n_class = 120):
    dtest = xgb.DMatrix(Xt.values)
    ymat = bst.predict(dtest)
    pclas = np.argmax(ymat, 1)    
    bins = np.linspace(0, 2*np.pi, n_class+1)
    clas = bins[0:-1] + np.diff(bins)/2
    Yp = clas[pclas]
    Yp = nap.Tsd(t = Xt.index.values, d = Yp, time_support = Xt.time_support)
    proba = nap.TsdFrame(t = Xt.index.values, d = ymat, time_support = Xt.time_support)
    return Yp, proba


def smoothAngle(angle, std):
    t = angle.index.values
    d = np.unwrap(angle.values)
    tmp = pd.Series(index = t, data = d)
    tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=std)
    tmp = tmp%(2*np.pi)
    try:
        tmp = nap.Tsd(tmp, time_support = angle.time_support)
    except:
        tmp = nap.Tsd(tmp)
    return tmp

def getAngularVelocity(angle, bin_size = None):
    dv = np.abs(np.diff(np.unwrap(angle.values)))
    dt = np.diff(angle.index.values)
    t = angle.index.values[0:-1]
    idx = np.where(dt<2*bin_size)[0]
    av = nap.Tsd(t=t[idx]+dt[idx]/2, d=dv[idx], time_support = angle.time_support)
    return av, idx

def plot_tc(tuning_curves, spikes):
    shank = spikes._metadata['group']
    figure()
    count = 1
    for j in np.unique(shank):
        neurons = np.array(spikes.keys())[np.where(shank == j)[0]]
        for k,i in enumerate(neurons):
            subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
            plot(tuning_curves[i], label = str(shank[i]) + ' ' + str(i))
            legend()
            count+=1
            gca().set_xticklabels([])
    show()

def centerTuningCurves_with_mean(tcurve, by=None):
    """
    center tuning curves by mean
    """
    if by is None:
        by = tcurve
    peak            = pd.Series(index=by.columns,data = np.array([circmean(by.index.values, by[i].values) for i in by.columns]))
    new_tcurve      = []
    for p in tcurve.columns:    
        x = tcurve[p].index.values - tcurve[p].index[np.searchsorted(tcurve[p].index, peak[p])-1]
        x[x<-np.pi] += 2*np.pi
        x[x>np.pi] -= 2*np.pi
        tmp = pd.Series(index = x, data = tcurve[p].values).sort_index()
        new_tcurve.append(tmp.values)
    new_tcurve = pd.DataFrame(index = np.linspace(-np.pi, np.pi, tcurve.shape[0]+1)[0:-1], data = np.array(new_tcurve).T, columns = tcurve.columns)
    return new_tcurve

def centerTuningCurves_with_peak(tcurve, by=None):
    """
    center tuning curves by peak index
    """
    if by is None:
        by = tcurve
    peak            = by.idxmax()
    new_tcurve      = []
    for p in tcurve.columns:    
        x = tcurve[p].index.values - tcurve[p].index[np.searchsorted(tcurve[p].index, peak[p])-1]
        x[x<-np.pi] += 2*np.pi
        x[x>np.pi] -= 2*np.pi
        tmp = pd.Series(index = x, data = tcurve[p].values).sort_index()
        new_tcurve.append(tmp.values)
    new_tcurve = pd.DataFrame(index = np.linspace(-np.pi, np.pi, tcurve.shape[0]+1)[0:-1], data = np.array(new_tcurve).T, columns = tcurve.columns)
    return new_tcurve
    


def compute_ISI_HD(spikes, angle, ep, bins):
    nb_bin_hd = 31
    tc2 = nap.compute_1d_tuning_curves(spikes, angle, nb_bin_hd, minmax=(0, 2*np.pi), ep = angle.time_support.loc[[0]])
    # angle2 = angle.restrict(ep)
    xbins = np.linspace(0, 2*np.pi, nb_bin_hd)
    xpos = xbins[0:-1] + np.diff(xbins)/2    

    pisiall = {}
    for n in spikes.keys():
        spk = spikes[n]
        isi = nap.Tsd(t = spk.index.values[0:-1]+np.diff(spk.index.values)/2, d=np.diff(spk.index.values))            
        idx = angle.index.get_indexer(isi.index, method="nearest")
        isi_angle = pd.Series(index = angle.index.values, data = np.nan)
        isi_angle.loc[angle.index.values[idx]] = isi.values
        isi_angle = isi_angle.fillna(method='ffill')
        isi_angle = nap.Tsd(isi_angle)        
        isi_angle = isi_angle.restrict(ep)

        # isi_angle = nap.Ts(t = angle.index.values, time_support = ep)
        # isi_angle = isi_angle.value_from(isi, ep)
        
        #data = np.vstack([np.hstack([isi_before.values, isi_after.values]), np.hstack([isi_angle.values, isi_angle.values])])
        data = np.vstack((isi_angle.values, angle.restrict(ep).values))

        pisi, _, _ = np.histogram2d(data[0], data[1], bins=[bins, xbins], weights = np.ones(len(data[0]))/float(len(data[0])))
        m = pisi.max()
        if m>0.0:
            pisi = pisi/m

        pisi = pd.DataFrame(index=bins[0:-1], columns=xpos, data=pisi)
        pisi = pisi.T
        # centering
        offset = tc2[n].idxmax()
        new_index = pisi.index.values - offset
        new_index[new_index<-np.pi] += 2*np.pi
        new_index[new_index>np.pi] -= 2*np.pi
        pisi.index = pd.Index(new_index)
        pisi = pisi.sort_index()
        pisi = pisi.T
        pisiall[n] = pisi

    return pisiall, xbins, bins

def decode_xgb(spikes, eptrain, bin_size_train, eptest, bin_size_test, angle, std = 1):    
    count = spikes.count(bin_size_train, eptrain)
    count = count.as_dataframe()    
    rate_train = count/bin_size_train
    #rate_train = rate_train.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
    rate_train = nap.TsdFrame(rate_train, time_support = eptrain)
    rate_train = zscore_rate(rate_train)
    rate_train = rate_train.restrict(eptrain)
    angle2 = getBinnedAngle(angle, angle.time_support.loc[[0]], bin_size_train).restrict(eptrain)

    count = spikes.count(bin_size_test, eptest)
    count = count.as_dataframe()
    rate_test = count/bin_size_test
    #rate_test = rate_test.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
    rate_test = nap.TsdFrame(rate_test, time_support = eptest)
    rate_test = zscore_rate(rate_test)

    angle_predi, proba, bst = xgb_decodage(Xr=rate_train, Yr=angle2, Xt=rate_test)

    return angle_predi, proba

def correlate_TC_half_epochs(spikes, feature, nb_bins, minmax):
    two_ep = splitWake(feature.time_support.loc[[0]])
    tcurves2 = []
    for j in range(2):
        tcurves_half = nap.compute_1d_tuning_curves(spikes, feature, nb_bins, minmax=minmax, ep = two_ep.loc[[j]])
        tcurves_half = smoothAngularTuningCurves(tcurves_half, 20, 3)
        tcurves2.append(tcurves_half)
    r = np.diag(np.corrcoef(tcurves2[0].values.T, tcurves2[1].values.T)[0:len(spikes), len(spikes):])
    r = pd.Series(index=list(spikes.keys()), data = r)
    return r

def read_neuroscope_intervals(path, basename, name):
    """
    """
    path2file = os.path.join(path, basename + "." + name + ".evt")
    # df = pd.read_csv(path2file, delimiter=' ', usecols = [0], header = None)
    tmp = np.genfromtxt(path2file)[:, 0]
    df = tmp.reshape(len(tmp) // 2, 2)
    isets = nap.IntervalSet(df[:, 0], df[:, 1], time_units="ms")
    return isets

def write_neuroscope_intervals(path, basename, name, isets):
    path2file = os.path.join(path, basename + "." + name + ".evt")

    start = isets.as_units("ms")["start"].values
    ends = isets.as_units("ms")["end"].values

    datatowrite = np.vstack((start, ends)).T.flatten()

    n = len(isets)

    texttowrite = np.vstack(
        (
            (np.repeat(np.array([name + " start"]), n)),
            (np.repeat(np.array([name + " end"]), n)),
        )
    ).T.flatten()
    
    f = open(path2file, "w")
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()


def decode_pytorch(spikes, eptrain, bin_size_train, eptest, bin_size_test, angle, std = 1):
    count = spikes.count(bin_size_train, eptrain)
    count = count.as_dataframe()
    rate_train = count/bin_size_train
    rate_train = rate_train.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
    rate_train = nap.TsdFrame(rate_train, time_support = eptrain)
    rate_train = zscore_rate(rate_train)
    rate_train = rate_train.restrict(eptrain)

    angle2 = getBinnedAngle(angle, angle.time_support.loc[[0]], bin_size_train).restrict(eptrain)

    numHDbins = 30
    HDbinedges = np.linspace(0,2*np.pi,numHDbins+1)
    HD_binned = np.digitize(angle2.values,HDbinedges)-1 #(-1 for 0-indexed category)    

    count = spikes.count(bin_size_test, eptest)
    count = count.as_dataframe()
    rate_test = count/bin_size_test
    rate_test = rate_test.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=std)
    rate_test = nap.TsdFrame(rate_test, time_support = eptest)
    rate_test = zscore_rate(rate_test)

    N_units = len(spikes)
    decoder = linearDecoder(N_units,numHDbins)

    #Train the decoder 
    batchSize = 0.75
    numBatches = 100000
    decoder.train(rate_train.values, HD_binned,
                batchSize=batchSize, numBatches=numBatches, Znorm=False)
    #%%
    #Decode HD from test set
    decoded, p = decoder.decode(rate_test.values)

    clas = HDbinedges[0:-1] + np.diff(HDbinedges)/2
    Yp = clas[decoded]
    Yp = nap.Tsd(t = rate_test.index.values, d = Yp, time_support = rate_test.time_support)

    return Yp, p


def loadOptoEp(path, epoch, n_channels = 2, channel = 0, fs = 20000):
    """
        load ttl from analogin.dat
    """
    files = os.listdir(path)
    afile = os.path.join(path, [f for f in files if '_'+str(epoch)+'_' in f][0])
    f = open(afile, 'rb')
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2        
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    f.close()
    with open(afile, 'rb') as f:
        data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
    data = data[:,channel].flatten().astype(np.int32)

    start,_ = scipy.signal.find_peaks(np.diff(data), height=3000)
    end,_ = scipy.signal.find_peaks(np.diff(data)*-1, height=3000)
    start -= 1
    timestep = np.arange(0, len(data))/fs
    # aliging based on epoch_TS.csv
    epochs = pd.read_csv(os.path.join(path, 'Epoch_TS.csv'), header = None)
    timestep = timestep + epochs.loc[epoch,0]
    opto_ep = nap.IntervalSet(start = timestep[start], end = timestep[end], time_units = 's')
    #pd.DataFrame(opto_ep).to_hdf(os.path.join(path, 'Analysis/OptoEpochs.h5'), 'opto')
    return opto_ep  

######################################################################################
# OPTO STUFFS
######################################################################################
def computeRasterOpto(spikes, opto_ep, bin_size = 100):
    """
    Bin size in ms
    edge in ms
    """
    rasters = {}
    frates = {}

    # assuming all opto stim are the same for a session
    stim_duration = opto_ep.loc[0,'end'] - opto_ep.loc[0,'start']
    
    bins = np.arange(0, stim_duration + 2*stim_duration + bin_size*1000, bin_size*1000)

    for n in spikes.keys():
        rasters[n] = []
        r = []
        for e in opto_ep.index:
            ep = nap.IntervalSet(start = opto_ep.loc[e,'start'] - stim_duration,
                                end = opto_ep.loc[e,'end'] + stim_duration)
            spk = spikes[n].restrict(ep)
            tmp = pd.Series(index = spk.index.values - ep.loc[0,'start'], data = e)
            rasters[n].append(tmp)
            count, _ = np.histogram(tmp.index.values, bins)
            r.append(count)
        r = np.array(r)
        frates[n] = pd.Series(index = bins[0:-1]/1000, data = r.mean(0))
        rasters[n] = pd.concat(rasters[n])      

    frates = pd.concat(frates, 1)
    frates = nap.TsdFrame(t = frates.index.values, d = frates.values, time_units = 'ms')
    return frates, rasters, bins, stim_duration

def computeLMNAngularTuningCurves(spikes, angle, ep, nb_bins = 180, frequency = 120.0, bin_size = 100):
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values)) 
    tmp2            = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)   
    bin_size        = bin_size * 1000
    time_bins       = np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
    index           = np.digitize(tmp2.index.values, time_bins)
    tmp3            = tmp2.groupby(index).mean()
    tmp3.index      = time_bins[np.unique(index)-1]+bin_size/2
    tmp3            = nap.Tsd(tmp3)
    tmp4            = np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
    newangle        = nap.Tsd(t = tmp3.index.values, d = tmp3.values%(2*np.pi))
    velocity        = nap.Tsd(t=tmp3.index.values[1:], d = tmp4)
    velocity        = velocity.restrict(ep) 
    velo_spikes     = {}    
    for k in spikes: velo_spikes[k] = velocity.realign(spikes[k].restrict(ep))
    # bins_velocity = np.array([velocity.min(), -2*np.pi/3, -np.pi/6, np.pi/6, 2*np.pi/3, velocity.max()+0.001])
    bins_velocity   = np.array([velocity.min(), -np.pi/6, np.pi/6, velocity.max()+0.001])

    idx_velocity    = {k:np.digitize(velo_spikes[k].values, bins_velocity)-1 for k in spikes}

    bins            = np.linspace(0, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    tuning_curves   = {i:pd.DataFrame(index = idx, columns = list(spikes.keys())) for i in range(3)}    

    # for i,j in zip(range(3),range(0,6,2)):
    for i,j in zip(range(3),range(3)):
        for k in spikes:
            spks            = spikes[k].restrict(ep)            
            spks            = spks[idx_velocity[k] == j]
            angle_spike     = newangle.restrict(ep).realign(spks)
            spike_count, bin_edges = np.histogram(angle_spike, bins)
            tmp             = newangle.loc[velocity.index[np.logical_and(velocity.values>bins_velocity[j], velocity.values<bins_velocity[j+1])]]
            occupancy, _    = np.histogram(tmp, bins)
            spike_count     = spike_count/occupancy 
            tuning_curves[i][k] = spike_count*(1/(bin_size*1e-6))

    return tuning_curves, velocity, bins_velocity

def computeLMN_TC(spikes, angle, ep, velocity):
    atitc = {}
    bins_velocity   = np.array([velocity.min(), -np.pi/12, np.pi/12, velocity.max()+0.001])
    # bins_velocity   = np.array([velocity.min(), -0.1, 0.1, velocity.max()+0.001])
    for n in spikes.index:
        vel = velocity.restrict(ep)
        spkvel = spikes[n].restrict(ep).value_from(velocity)
        idx = np.digitize(spkvel.values, bins_velocity)-1
        tcvel = {}
        for k in range(3):
            spk = nap.TsGroup({0:nap.Tsd(spkvel.as_series().iloc[idx == k], time_support = ep)})
            tc = nap.compute_1d_tuning_curves(spk, angle, 120, minmax=(0, 2*np.pi), ep = ep)
            tc = smoothAngularTuningCurves(tc, 50, 2)
            tcvel[k] = tc[0]
        tcvel = pd.DataFrame.from_dict(tcvel)
        atitc[n] = tcvel
    return atitc


def downsample(tsd, up, down):
    import scipy.signal
    dtsd = scipy.signal.resample_poly(tsd.values.astype(np.float32), up, down)
    dt = tsd.index.values[np.arange(0, tsd.shape[0], down)]
    return nap.Tsd(t=dt, d=dtsd, time_support = tsd.time_support)


def load_data(filepath):

    # basename = os.path.basename(path)
    # filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    nwb = nap.load_file(filepath)
    spikes = nwb['units']
    spikes = spikes.getby_threshold("rate", 1)

    position = []
    columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    for k in columns:
        if k == 'ry':
            ry = nwb[k].values[:]
            position.append((ry + np.pi)%(2*np.pi))
        else:
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
    nwb.close()

    
    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
    tuning_curves = smoothAngularTuningCurves(tuning_curves)
    tcurves = tuning_curves
    SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
    spikes.set_info(SI)
    spikes.set_info(max_fr = tcurves.max())
    
    spikes = spikes.getby_threshold("rate", 1.0)
    spikes = spikes.getby_threshold("max_fr", 3.0)

    tokeep = spikes.index
    tcurves = tcurves[tokeep]

    # peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
    peaks = tcurves.idxmax()
    order = np.argsort(peaks.reset_index(drop='True').sort_values().index)
    spikes.set_info(order=order, peaks=peaks)

    return spikes, position, {'wake_ep':wake_ep, 'sws_ep':sws_ep}

def load_opto_data(path, st):
    dropbox_path = os.path.join(os.path.expanduser("~"), "Dropbox/LMNphysio/data/" + os.path.basename(path) + ".pickle")
    if os.path.exists(dropbox_path):
        import _pickle as cPickle
        data = cPickle.load(open(dropbox_path, 'rb'))
        spikes = data['spikes']
        position = data['position']
        wake_ep = data['wake_ep']
        opto_ep = data['opto_ep']
        sws_ep = data['sws_ep']

    else:    
        SI_thr = {
            'adn':0.1, 
            'lmn':0.1,
            'psb':0.1
            }    
        basename = os.path.basename(path)
        filepath = os.path.join(path, "kilosort4", basename + ".nwb")
        nwb = nap.load_file(filepath)
        spikes = nwb['units']
        spikes = spikes.getby_threshold("rate", 8)

        position = []
        columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        for k in columns:
            if k == 'ry':
                ry = nwb[k].values[:]
                position.append((ry + np.pi)%(2*np.pi))
            else:
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
        sws_ep = nwb['sws']
        nwb.close()

        spikes = spikes[spikes.location == st]        
        stim_duration = 1.0
        opto_ep = opto_ep[(opto_ep['end'] - opto_ep['start'])>=stim_duration-0.001]

        tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        tuning_curves = smoothAngularTuningCurves(tuning_curves)
        tcurves = tuning_curves
        SI = nap.compute_1d_mutual_info(tcurves, position['ry'], position.time_support.loc[[0]], (0, 2*np.pi))
        spikes.set_info(SI)
        spikes.set_info(max_fr = tcurves.max())

        spikes = spikes.getby_threshold("SI", SI_thr[st])
        spikes = spikes.getby_threshold("rate", 1.0)
        spikes = spikes.getby_threshold("max_fr", 3.0)

        tokeep = spikes.index
        tcurves = tcurves[tokeep]

        # peaks = pd.Series(index=tcurves.columns, data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns]))
        peaks = tcurves.idxmax()
        order = np.argsort(peaks.reset_index(drop='True').sort_values().index)
        spikes.set_info(order=order, peaks=peaks)

        # Adding to dropbox
        datatosave = {"spikes":spikes, "position":position, 'wake_ep':wake_ep, 'opto_ep':opto_ep, 'sws_ep':sws_ep}
        savepath = os.path.join(os.path.expanduser("~"), "Dropbox/LMNphysio/data/" + os.path.basename(path) + ".pickle")
        import _pickle as cPickle
        cPickle.dump(datatosave, open(savepath, 'wb'))


    return spikes, position, wake_ep, opto_ep, sws_ep

import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy.signal.windows import dpss
import pandas as pd


def getfgrid(Fs, nfft, fpass):
	"""
		get frequency grid for evaluation
	"""
	df			= Fs/nfft
	f 			= np.arange(0, Fs+df, df)
	f 			= f[0:nfft]
	findx		= np.logical_and( f>=fpass[0],  f<=fpass[-1] )
	f 			= f[findx]
	return f, findx

def dpsschk(tapers, N, Fs):
	"""
		check tapers
	"""
	tapers, eigs = dpss(N, NW = tapers[0], Kmax=tapers[1], sym=False, return_ratios = True)
	tapers 		= tapers * np.sqrt(Fs)
	tapers 		= tapers.T
	return tapers

def mtfftpt(data, tapers, nfft, t, f, findx):
	"""
		mt fft for point process times
	"""
	C 			= 1
	K 			= tapers.shape[1]
	nfreq		= len(f)
	H 			= np.zeros((nfft,tapers.shape[1]), dtype = np.complex128)
	for i in range(tapers.shape[1]):
		H[:,i] 		= fft(tapers[:,i], nfft)

	H 			= H[findx,:]
	w 			= 2*np.pi*f
	dtmp		= data
	indx 		= np.logical_and(dtmp >= np.min(t), dtmp <= np.max(t))
	if len(indx): dtmp = dtmp[indx]
	Nsp			= len(dtmp)
	Msp 		= Nsp/len(t)

	if Msp != 0:
		data_proj = []
		for i in range(tapers.shape[1]):
			ff = interp1d(t, tapers[:,i])
			data_proj.append(ff(dtmp))
		data_proj = np.array(data_proj).T
		exponential = np.exp(np.atleast_2d(-1j * w).T * (dtmp - t[0]))
		J = np.dot(exponential, data_proj) - H*Msp
	else:
		J = np.zeros((nfreq, tapers.shape[1]))

	return J, Msp, Nsp

def mtspectrumpt(data, Fs, fpass, tapers, trialave = 1, err = [1, 0.05], pad = -1):
	""" 
		mtspectrumpt from chronux toolbox
	"""
	mintime 	= np.min(data)
	maxtime 	= np.max(data)
	dt 			= 1/Fs
	t 			= np.arange(mintime-dt, maxtime+2*dt, dt)
	N 			= len(t)
	nfft 		= int(np.maximum(2**(np.ceil(np.log2(np.abs(N))) + pad), N)) # number of points in fft of prolates
	f, findx 	= getfgrid(Fs, nfft, fpass)
	tapers 		= dpsschk(tapers, N, Fs)
	J,Msp,Nsp	= mtfftpt(data,tapers,nfft,t,f,findx)
	S = np.real(np.mean(np.conj(J)*J, 1))
	return pd.Series(index = f, data = S)







# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-02-22 10:50:11
# @Last Modified by:   gviejo
# @Last Modified time: 2022-02-22 10:51:14

import numpy as np
import pandas as pd
import sys, os
import scipy
from scipy import signal
from itertools import combinations


def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
	new_tuning_curves = {}	
	for i in tuning_curves.columns:
		tcurves = tuning_curves[i]
		offset = np.mean(np.diff(tcurves.index.values))
		padded 	= pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi)-offset,
												tcurves.index.values,
												tcurves.index.values+(2*np.pi)+offset)),
							data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
		smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)		
		new_tuning_curves[i] = smoothed.loc[tcurves.index]

	new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

	return new_tuning_curves

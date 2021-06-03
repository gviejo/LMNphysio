import numpy as np
from pylab import *
import pandas as pd


file = '/mnt/DataAdrienBig/PeyracheLabData/Sofia/A7801-210226-2-Exp-Camera 5 (#378042)DLC_mobnet_100_SofiaRescueTheSecondMar3shuffle1_300000.csv'


data = pd.read_csv(file, header = [1,2])
data = data.drop(labels = 1, axis = 1)

front = data['bodypart1'][['x', 'y']]
left = data['bodypart2'][['x', 'y']]
right = data['bodypart3'][['x', 'y']]

back = pd.concat(
		(pd.concat((left['x'],right['x']),1).mean(1),
		pd.concat((left['y'],right['y']),1).mean(1)),1
		)
back.columns = ['x', 'y']

df = front - back
angle = np.arctan2(df['y'], df['x'])

angle[angle<0] += 2*np.pi
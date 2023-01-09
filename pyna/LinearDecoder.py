#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:33:40 2022

@author: dl2820
"""
import torch
from torch import nn
import numpy as np


class linearDecoder:
    def __init__(self, numUnits, numX):
        """
        
        Parameters
        ----------
        numUnits : Number of units
        numX : Number of spatial locations

        """
        self.model = nn.Linear(numUnits, numX, bias=False)
        
        #Pick the optimizer. Adam is usually a good choice.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3,
                                           weight_decay=3e-1)
        
        #Cross Entropy Loss - loss between vector (argmax) and categorical data
        self.loss_fn = nn.CrossEntropyLoss()


    def decode(self, h, withSoftmax=True, asNumpy=True):
        """
        Parameters
        ----------
        h : [Nt x Nunits] numpy array
        withSoftmax : T/F, optional. 
            If true, returns probability in each bin. The default is True.
        asNumpy : T/F, optional.
            If True, outputs are numpy. If False, pytorch tensor.

        Returns
        -------
        decodedX : numpy array or pytorch tensor
            The decoded location.
        p_X : numpy array or pytorch tensor
            Probabilty of being at each location.

        """
        #Normalize h
        h = (h-self.h_mean)/self.h_std
        #Convert h to pytorch tensor
        h = torch.tensor(h, dtype=torch.float)
        
        p_X = self.model(h)
        decodedX = p_X.argmax(dim=1)
        
        if withSoftmax:
            sm = nn.Softmax(dim=1)
            p_X = sm(p_X)
            
        if asNumpy:
            decodedX = decodedX.detach().numpy()
            p_X = p_X.detach().numpy()
            
        return decodedX, p_X


    def train(self, h, pos, batchSize=0.75, numBatches = 10000,
              Znorm=False):
        """
        Train the decoder from activity-position pairs

        Parameters
        ----------
        h : [Nt x Nunits] numpy array of neural data 
        pos : [Nt] numpy array of behavioral data
            The (binned/linearized) spatial position at each timestep
        batchSize : optional
            Fraction of data to use each learning step. Default: 0.75.
        numBatches : optional
            How many training steps. Default: 10000

        """

        #If Znorm: subtract mean and divide by STD. Save the norm values
        if Znorm:
            self.h_mean = np.mean(h,axis=0)
            self.h_std = np.std(h,axis=0)
        else:
            self.h_mean = np.zeros(h.shape[1])
            self.h_std = np.ones(h.shape[1])
        
        #Consider: while loss doesn't change or is big enough...
        print('Training Decoder...')
        for step in range(numBatches): 
            batch = np.random.choice(pos.shape[0],int(batchSize*pos.shape[0]),replace=False)
            h_batch,pos_batch = h[batch,:],pos[batch]
            steploss = self.trainstep(h_batch,pos_batch)
            if (100*step /numBatches) % 10 == 0 or step==numBatches-1:
                print(f"loss: {steploss:>f} [{step:>5d}\{numBatches:>5d}]")
        return
    
    
    def trainstep(self,h_train,pos_train):
        """
        One training step of the decoder. 
        (Should only be called by linearDecoder.train)
        """
        decodedX,p_X = self.decode(h_train, withSoftmax=False, asNumpy=False)
        
        pos_train = torch.tensor(pos_train)
        loss = self.loss_fn(p_X,pos_train)
        
        self.optimizer.zero_grad()   #Reset the gradients
        loss.backward()              #Backprop the gradients w.r.t loss
        self.optimizer.step()        #Update parameters one step
        
        steploss = loss.item()
        return steploss




###############################################################################
##The Linear Model#############################################################
###############################################################################

# class linnet(nn.Module):
#     def __init__(self,numUnits,numX):
#         super(linnet, self).__init__()
#         self.lin = nn.Linear(numUnits, numX, bias=False)
        
#     def forward(self,x):
#         logits = self.lin(x)
#         return logits
    
    
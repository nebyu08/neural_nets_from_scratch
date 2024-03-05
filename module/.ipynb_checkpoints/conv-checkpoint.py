from .base import paramoperation
from __future__ import division
import numpy as np

class conv_2d(paramoperation):
    def __init__(self,w:np.ndarray):
        ''' w/filter has the shape input channel,output channel,heigh and width'''
        super().__init__(w)
        self.param_size=w.shape[2]
        self.param_pad=self.param_size//2
    
    def _pad_1d(self,inp:np.ndarray)->np.ndarray:
        #this is one dimensional padding
        z=np.array([0])
        z=np.repeat(z,self.param_pad)
        return np.concatenate([z,inp,z])
        
    def _pad_1d_batch(self,inp:np.ndarray)->np.ndarray:
        outs=[_pad_1d(obs) for obs in inp]
        return np.stack(outs)
        
    def _pad_2d_obs(self,inp:np.ndarray):
        inp_pad=self._pad_1d_batch(inp)
        temp=np.zeros((self.param,inp.shape[0]+self.param_pad*2))  #this is the zeros that go into the batch of input
        return np.conctatenate(temp,inp_pad,temp)
        
    def _pad_2d_channels(self,inp:np.ndarray)->np.ndarray:
        ''' the dimension of inp is [num channels,img width and img height'''
        return np.stack([self._pad_2d_obs(channel) for channel in inp])
from .base import operation
import numpy as np

class DropOut(operation):
    ''' this drops neurons randomly '''
    def __init__(self,keep_prob):
        self.keep_prob=keep_prob
        super().__init__()
        
    def _output(self)->np.ndarray:
        if self.inference:
           return self.input_*self.keep_prob
        else:
            self.mask=np.random.binomial(1,self.keep_prob,size=self.input_.shape)
            return self.input_*self.mask
    def _input_grad(self,output_grad)->np.ndarray:
        ''' this is during backpropagation'''
        
        return output_grad*self.mask
       
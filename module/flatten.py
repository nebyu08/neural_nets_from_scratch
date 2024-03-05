from .base import operation
import numpy as np

class Flatten(operation):
    ''' this flattens/squashes the 3d input into 1d'''
    def __init__(operation):
        super().__init__()
        
    def _output(self):
        return self._input.reshape(self._input.shape[0],-1)
        
    def _input_grad(self,output_grad:np.ndarray)->np.ndarray:
        return output_grad.reshape(self._input.shape)
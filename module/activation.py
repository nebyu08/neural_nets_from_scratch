from .base import operation
import numpy as np

class linear(operation):
    ''' this is the linear activation function'''
    def __init__(self):
        super().__init__()
    def _output(self):
        return self.input_
    def _input_grad(self,output_grad:np.ndarray):
        return output_grad

class sigmoid(operation):
    '''this is sigmoid function'''
    def __init__(self):
        super().__init__()
        
    def _output(self)->np.ndarray:
        return 1.0/(1.0+np.exp(-1.0*self.input_))
        
    def _input_grad(self,output_grad:np.ndarray)->np.ndarray:
        ''' this is the input grad with respect to the sigmoid fucntion'''
        sigmoid_back=self.output*(1-self.output)
        input_grad=sigmoid_back*output_grad
        return input_grad
        
class tanh(operation):
    def __init__(self):
        super().__init__()
    def _output(self)->np.ndarray:
        return np.tanh(self.input_)
    def _input_grad(self,output_grad)->np.ndarray:
        return output_grad*(1-self.output*self.output)
        
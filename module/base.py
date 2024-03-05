import numpy as np
from .utils import assert_same_shape

class operation(object):
    def __init__(self):
        pass
        
    def forward(self,input_:np.ndarray,inference:bool=False)->np.ndarray:
        self.input_=input_
        self.inference=inference
        self.output=self._output()
        return self.output

    def backward(self, output_grad):
        ''' here we calculate the input gradient after passing the output grad'''
        assert_same_shape(self.output,output_grad)
        self.input_grad=self._input_grad(output_grad)
        assert_same_shape(self.input_,self.input_grad)
        return self.input_grad
        
    def _output(self):
        raise NotImplementedError()
    
    def _input_grad(self,output_grad:np.ndarray):
        raise NotImplementedError()
        
class paramoperation(operation):
    def __init__(self,param: np.ndarray):
        super().__init__()
        self.param=param
    def backward(self,output_grad:np.ndarray):
        assert_same_shape(self.output,output_grad)
        self.input_grad=self._input_grad(output_grad)
        self.param_grad=self._param_grad(output_grad)
        assert_same_shape(self.input_,self.input_grad)
        return self.input_grad
    def _input_grad(self,output_grad:np.ndarray):
        raise NotImplementedError()
    def _param_grad(self,output_grad:np.ndarray):
        raise NotImplementedError()
        

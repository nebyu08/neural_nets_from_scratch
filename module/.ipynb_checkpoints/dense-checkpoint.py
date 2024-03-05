from .base import paramoperation
import numpy as np

class weightmultiply(paramoperation):
    ''' this child calculates the weight multiplication of the backward forward NN'''
    def __init__(self,param:np.ndarray):
        super().__init__(param)
    def _output(self):
        return np.dot(self.input_,self.param)
    def _input_grad(self,output_grad:np.ndarray):
        return np.dot(output_grad,np.transpose(self.param,(1,0)))
    def _param_grad(self,output_grad:np.ndarray):
        return np.dot(np.transpose(self.input_,(1,0)),output_grad)
        
class biassadd(paramoperation):
    ''' performs the biass backpropagation '''
    def __init__(self,biass:np.ndarray):
        super().__init__(biass)
        
    def _output(self)->np.ndarray:
        return self.input_ + self.param
    def _input_grad(self,output_grad:np.ndarray)->np.ndarray:
        return np.ones_like(self.input_)*output_grad
    def _param_grad(self,output_grad:np.ndarray):
        ''' this is derivative is with respect to the biass'''
        output_grad_reshaped=np.sum(output_grad,axis=0).reshape(1,-1)  # the summation is along the row
        param_grad=np.ones_like(self.param)
        return param_grad*output_grad_reshaped
from .utils import assert_same_shape,softmax
import numpy as np

class Loss(object):
    ''' this is the losses for the neural network'''
    def __init__(self):
        pass
        
    def forward(self,target:np.ndarray,
                prediction:np.ndarray):
        # lets assert the same shape
        assert_same_shape(target,prediction)
        
        self.target=target
        self.prediction=prediction
        self.output=self._output()  # the calculated loss
        
        return self.output

    def backward(self):
        self.input_grad=self._input_grad()
        assert_same_shape(self.input_grad,self.prediction)
        return self.input_grad
        
    def _output(self):
         raise NotImplementedError()
        
    def _input_grad(self):
        raise NotImplementedError()
        
class meansquarederror(Loss):
    def __init__(self,normalize:bool=False):
        super().__init__()
        self.normalize=normalize
    def _output(self):
        ''' this calculates the actuall loss it self'''
        if self.normalize:
            self.prediction=self.prediction/np.sum(self.prediction,axis=1,keepdims=True) #making it in a propabilistic format
        loss=np.sum(np.power(self.prediction-self.target,2))/self.prediction.shape[0]
        return loss
    def _input_grad(self):
        return 2*(self.prediction- self.target)/self.prediction.shape[0]
        
class softmaxcrossentropy(Loss):
    def __init__(self,eps:float=1e-9):
        super().__init__()
        self.eps=eps
        self.single_output=False
    def _output(self):
        #lets apply the softmax
        softmax_preds=softmax(self.prediction,axis=1)
        self.softmax_preds=np.clip(softmax_preds,self.eps,1-self.eps)
        #lets make the softmax cross entropy
        softmax_cross_entropy=(-1.0*self.target*np.log(self.softmax_preds)-(1-self.target)*np.log(1-self.softmax_preds))
        return np.sum(softmax_cross_entropy)
    def _input_grad(self):
        return self.softmax_preds-self.target
        
from typing import List
from .layer import Layers
from .losses import Loss,meansquarederror
import numpy as np

class LayerBlock(object):
    ''' simply make the forward pass and backward pass on the layers
      and alsot this is basically the layer making class...'''
    
    def __init__(self,layers:List[Layers]):
        super().__init__()
        self.layers=layers
        
    def forward(self,x_batch:np.ndarray,inference:bool=False):
        x_out=x_batch
        for layer in self.layers:
            x_out=layer.forward(x_out,inference)
        return x_out
        
    def backward(self,loss_grad:np.ndarray):
        grad=loss_grad
        for layer in reversed(self.layers):
            grad=layer.backward(grad)
        return grad

    def params(self):
        for layer in self.layers:
            yield from layer.params
            
    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads
    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"


class NeuralNetwork(LayerBlock):
    ''' a wrapper class that contains the rest of layers'''
    def __init__(self,
                layers:List[Layers],
                loss:Loss=meansquarederror,
                seed:int=34
                ):
        super().__init__(layers)
        self.loss=loss
        self.seed=seed
        if seed:
            for layer in self.layers:
                setattr(layer,"seed",self.seed)  #setting as attribute of layers
                
    def forward_loss(self,
                     x_batch:np.ndarray,
                     y_batch:np.ndarray,
                     inference:bool=False):
        
        prediction=self.forward(x_batch,inference)
        return self.loss.forward(y_batch,prediction)
        
    def train_batch(self,
                    x_batch:np.ndarray,
                    y_batch:np.ndarray,
                    inference:bool=False
                   )->float:

        
        prediction=self.forward(x_batch,inference)
        batch_loss=self.loss.forward(y_batch,prediction)
        loss_grad=self.loss.backward()
        self.backward(loss_grad)
        
        return batch_loss
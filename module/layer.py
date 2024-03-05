import numpy as np
from .utils import assert_same_shape
from typing import Tuple,Dict,List
from .base import operation,paramoperation
from .dense import weightmultiply,biassadd
from .activation import linear
from .dropout import DropOut

class Layers(object):
    
    ''' this is a layers in N.N which is basically a wrapper function that containes the operation class '''
    
    def __init__(self,neurons:int)->None:
        self.neurons=neurons
        self.first=True
        self.params:List[np.ndarray]=[]
        self.param_grads:List[np.ndarray]=[]
        self.operations:List[operation]=[]

    
    def _setup_layer(self,input_:np.ndarray)->None:
        pass


    def forward(self,input_: np.ndarray,inference=False)->np.ndarray:
        ''' this is the feed forward function part of the deep nerual network'''
        if self.first:
            self._setup_layer(input_)  #this is the weight initializer
            self.first=False
        self.input_=input_
        for index,operation in enumerate(self.operations):
            try:
                input_=operation.forward(input_,inference) 
            except TypeError as e:
                print(f"the error is in {index+1} operation and the error is {e}")
                raise
        self.output=input_
        #print(f"the outputs shape is {input_.shape}")
        return self.output
        
    def backward(self,output_grad:np.ndarray):
        ''' this is the backpropagation of the layer'''
        assert_same_shape(self.output,output_grad)
        for operation in self.operations[::-1]:  # this operation start at the end of the list and goes backwards to the beginging
            output_grad=operation.backward(output_grad)
        input_grad=output_grad
        assert_same_shape(self.input_,input_grad)
        self._params_grads()  #for storing the parameter gradients
        return input_grad
        
    def _params_grads(self):
        self.param_grads=[]
        for operation in self.operations:
            if issubclass(operation.__class__,paramoperation):
                self.param_grads.append(operation.param_grad)
        
        
    def _params(self):
        self.params=[]
        for operation in self.operations:
            if issubclass(operation.__class__,paramoperation):
                self.params.append(operation.param)

class Dense(Layers):
    ''' this operation only contains the weight initialization technique for the weight'''
    
    def __init__(self,
                 neurons:int,
                 activation:operation=linear(),
                 weight_init:str="standard",
                 dropout:float =0.2
                )->None:
        
        super().__init__(neurons)
        self.activation=activation
        self.weight_init=weight_init
        self.dropout=dropout
             

    def _setup_layer(self,input_:np.ndarray)->None:

        self.input_=input_
        np.random.seed(self.seed)

        num_in=input_.shape[1]
        
        if self.weight_init=="glorot":
            scale=np.sqrt(2/(num_in+self.neurons))
        
        else:
            scale=1.0
        
        #weights
        self.params=[]
        self.params.append(np.random.normal(loc=0,scale=scale,size=(num_in,self.neurons)))
        
        #biases
        self.params.append(np.random.normal(loc=0,scale=scale,size=(1,self.neurons)))

        #this is basically the operation of the "Layers"
        self.operations=[weightmultiply(self.params[0]),
                         biassadd(self.params[1]),
                         self.activation]
        
        if self.dropout<1.0:
            self.operations.append(DropOut(self.dropout))
            
        return None
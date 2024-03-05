import numpy as np

class optimizer(object):
    ''' this is for optimizing the weights of the neurla network '''
    def __init__(self,
                 lr:float=0.1,
                final_lr:float =0,
                decay_type:str =None):
        '''we use learning rate specifying the rate at which the model is going to be build'''
        self.lr=lr
        self.first=True
        self.final_lr=final_lr
        self.decay_type=decay_type
    def step(self):
        '''this is for calculating or implmenting of the optimizing'''
        pass
        
    def _setup_decay(self):
        ''' classifies the type of decay type to use'''
        if not self.decay_type:
            return 
        elif self.decay_type=="Exponential":
            self.decay_per_epoch=np.power(self.final_lr/self.lr,1.0/(self.max_epochs-1))
            
        elif self.decay_type=="linear":
            self.decay_per_epoch=(self.lr-self.final_lr)/(self.max_epochs-1)
            
    def _decay_lr(self):
        ''' decaies the learning rate'''
        if not self.decay_type==None:
            return 
        elif self.decay_type=="Exponential":
            self.lr*self.decay_per_epoch
        elif self.decay_type=="linear":
            self.lr*self.decay_per_epoch
    

class SGD(optimizer):
    ''' this optimizer is the stochastic gradient decent'''
    def __init__(self,lr:float=0.01)->None:
        super().__init__(lr)
        
    def step(self):
        '''we adjust the parameters based on the learning rate of the model'''
        for (param,param_grad) in zip(self.net.params(),self.net.param_grads()):
            param-=self.lr*param_grad
            
class SGDMomentum(optimizer):
    '''updates the parameters but first saves the parameter'''
    
    def __init__(self,
                 lr:float=0.01,
                final_lr:float=0.1,
                 momentum:float=0.9,
                decay_type:str=None):
        
        super().__init__(lr,final_lr,decay_type)
        self.momentum=momentum
        self.lr=lr
        
    def step(self):
        if self.first:
            self.history=[np.ones_like(param) for param in self.net.param_grads()]
            self.first=False
            
        for (param,param_grads,history) in zip(self.net.params(),self.net.param_grads(),self.history):
            self._update_rule(param=param,grads=param_grads,history=history)
            
    def _update_rule(self,**kwargs):
        kwargs["history"]*=self.momentum
        kwargs["history"]+=self.lr*kwargs["grads"]
        kwargs["param"]-=kwargs["history"]
        
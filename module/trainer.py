from .network import NeuralNetwork
from .optimizers import optimizer
from typing import Tuple
from copy import deepcopy
from .utils import permute_data
import numpy as np

class Trainer(object):
    ''' this simply trains the nerual network that we already defined '''
    def __init__(self,
                net:NeuralNetwork,
                optim:optimizer)->None:
        self.net=net
        self.optim=optim
        self.best_loss=1e9  # made it super big on purpose
        setattr(self.optim,"net",self.net)   # adding attribute to self.optim
        
    def fit(self,x_train:np.ndarray,y_train:np.ndarray,
           x_test:np.ndarray,y_test:np.ndarray,epochs:int=100,
           eval_every:int=10,batch_sz:int=32,seed:int=42,
            restart:bool=False,early_stoping:bool=True)->None:
        np.random.seed(seed)

        if restart:
            ''' this is for the case the nerual nets need to be restarted to their initial value'''
            for layer in self.net.layers:
                layer.first=True
            self.best_loss=1e9
            
        #training time
        for i in range(epochs):
            
            if (i+1)%eval_every==0:
                last_model=deepcopy(self.net)  #copying the N.N
                
            x_train,y_train=permute_data(x_train,y_train)
            batch_generator=self.generate_batches(x_train,y_train,batch_sz)
            
            for ii,(x_batch,y_batch) in enumerate(batch_generator):
                self.net.train_batch(x_batch,y_batch)
                self.optim.step()
            #lets check for whether its evaluation time....this is know inference mode
            if (i+1)%eval_every==0:
                test_preds=self.net.forward(x_test,inference=True)
                loss=self.net.loss.forward(y_test,test_preds)  #this is the test/validation loss 
                
                if early_stoping:
                    if loss<self.best_loss:
                        print(f"validation loss after {i+1} is {loss:.3f}")
                        self.best_loss=loss
                    else:
                        print()
                        print(f"loss increased after {i+1},final loss was {self.best_loss:.3f},",
                             f"\n we are using the model from epoch {i+1-eval_every}")
                        self.net=last_model
                        setattr(self.optim,"net",self.net)
                        break
                else:
                    print(f"validation loss after {i+1} epoch is {loss:.3f}")
            
            if self.optim.final_lr:
                self.optim._decay_lr()
                
        print("training complete")
        
    def generate_batches(self,x:np.ndarray,y:np.ndarray,batch_sz:int=32)->Tuple[np.ndarray]:
        
        assert x.shape[0] == y.shape[0],\
        ''' number of features and instances must have the same number of rows instead 
        x has {0} and y has {1} '''.format(x.shape[0],y.shape[0])
        
        n=x.shape[0] #this is the number of total instances
        
        for i in range(0,n,batch_sz):
            x_batch,y_batch=x[i:i+batch_sz],y[i:i+batch_sz]
            yield x_batch,y_batch
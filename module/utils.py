import numpy as np
from scipy.special import logsumexp
from typing import Tuple,Dict

def assert_same_shape(output:np.ndarray,
                     output_grad:np.ndarray)->None:
    assert output.shape == output_grad.shape ,\
    ''' there is a shape mismatch the first one is {0}
    while the second is {1}'''.format(tuple(output.shape),tuple(output_grad.shape))
    return None

def softmax(x:np.ndarray,axis=None):
    return np.exp(x-logsumexp(x,axis=axis,keepdims=True))

def normalize(a:np.ndarray):
    ''' this concatinated an array and its compliment'''
    comp=1-a
    return np.concatinate([a,comp],axis=1)

def unnormalize(a:np.ndarray):
    ''' this return the first row of the 2D array as a 2D array'''
    return a[np.newaxis,0]

def permute_data(x:np.ndarray,y:np.ndarray):
    perm=np.random.permutation(x.shape[0])
    return x[perm],y[perm]

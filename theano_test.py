__author__ = 'cclamb'


import numpy as np
import theano.tensor as T

from theano import function

def cell():
    value = T.scalar('value')
    weight = T.scalar('weight')
    transform = T.dmatrix('transform')
    return function([value, weight, transform], value * weight * transform)

class RouterCell(object):

    def __init__(self, weight=1.0, transformation=np.array([[0, 1], [0, 0]])):
        return

class Router(object):

    def __init__(self):
        return

class Processor(object):

    def __init__(self):
        return



__author__ = 'cclamb'


import numpy as np
import theano.tensor as T

from theano import function


class Router(object):

    def __init__(self):
        self.values = [
            [
                (1, np.array
                    (
                        [
                            [[1, 0],
                             [0, 0]],
                            [[0, 0],
                             [0, 1]]
                        ]
                    )
                 ),
                (2, np.array
                    (
                        [
                            [[1, 0],
                             [0, 0]],
                            [[0, 1],
                             [0, 0]]
                        ]
                    )
                )
            ],
            [
                (2.3, np.array
                    (
                        [
                            [[0, 0],
                             [0, 1]],
                            [[1, 0],
                             [0, 0]]
                        ]
                    )
                ),
                (3.2, np.array
                    (
                        [
                            [[0, 0],
                             [0, 0]],
                            [[0, 1],
                             [0, 0]]
                        ]
                    )
                ),
            ]
        ]
        value = T.scalar('value')
        weight = T.scalar('weight')
        transform = T.dmatrix('transform')
        self.mapping = function([value, weight, transform], value * weight * transform)
        return

    def map(self, values):
        result = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                # import ipdb
                # ipdb.set_trace()
                value = values[i][j]
                weight = self.values[i][j][0]
                transform = self.values[i][j][1]
                result[i][j] = self.mapping(
                    value,
                    weight,
                    transform.reshape(2, 4)
                ).reshape(2, 2, 2)
        return result


class Memory(object):

    def __init__(self):
        self.memory = np.array(
            [
                [[0, 0],
                 [0, 0]],
                [[0, 0],
                 [0, 0]]
            ]
        )
        memory = T.dmatrix('memory')
        value = T.dmatrix('value')
        self.mapping = function([memory, value], memory + value)

    def insert(self, value):
        self.memory = self.mapping(
            value.reshape(2, 4),
            self.memory.reshape(2, 4)
        ).reshape(2, 2, 2)


class Pipeline(object):

    def __init__(self):
        self.router = Router()
        self.memory = Memory()

    def run(self, values):
        mapping = self.router.map(values)
        for i in range(2):
            for j in range(2):
                self.memory.insert(mapping[i][j])
        return self.memory.memory

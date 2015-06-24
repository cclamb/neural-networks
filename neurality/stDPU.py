__author__ = 'cclamb'

import numpy as np
import theano.tensor as T

from theano import function


class Router(object):
    def __init__(self, values):
        self.values = values
        # self.values = [
        #     [
        #         (1, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          ),
        #         (2, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          ),
        #         (2, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          )
        #     ],
        #     [
        #         (2.3, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          ),
        #         (3.2, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          ),
        #         (2, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          )
        #     ],
        #     [
        #         (2.3, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          ),
        #         (3.2, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          ),
        #         (2, np.array
        #             (
        #             [
        #                 [[1, 0, 0],
        #                  [0, 0, 1],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]],
        #                 [[0, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 0]]
        #             ]
        #         )
        #          )
        #     ]
        # ]
        value = T.scalar('value')
        weight = T.scalar('weight')
        transform = T.dmatrix('transform')
        self.mapping = function([value, weight, transform], value * weight * transform)
        return

    def map(self, values):
        result = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        for i in range(3):
            for j in range(3):
                value = values[i][j]
                weight = self.values[i][j][0]
                transform = self.values[i][j][1]
                # import ipdb
                # ipdb.set_trace()
                result[i][j] = self.mapping(
                    value,
                    weight,
                    transform.reshape(3, 9)
                ).reshape(3, 3, 3)
        return result


class Memory(object):
    def __init__(self):
        self.memory = np.zeros((3, 3, 3))
        memory = T.dmatrix('memory')
        value = T.dmatrix('value')
        self.mapping = function([memory, value], memory + value)

    def insert(self, value):
        self.memory = self.mapping(
            value.reshape(3, 9),
            self.memory.reshape(3, 9)
        ).reshape(3, 3, 3)


class Integrator(object):
    def __init__(self):
        self.threshold = 0
        return

    def integrate(self, memory):
        # result = [
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        # ]
        result = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                sum = 0
                for k in range(3):
                    sum += memory[i][j][k] + self.threshold
                result[i][j] = sum
        return result


class Pipeline(object):

    def __init__(self, router_values):
        self.router = Router(router_values)
        self.memory = Memory()
        self.integrator = Integrator()

    def run(self, values):
        mapping = self.router.map(values)
        for i in range(3):
            for j in range(3):
                self.memory.insert(mapping[i][j])
        return self.integrator.integrate(self.memory.memory)


class Combinator(object):

    def combine(self, lhs, rhs):
        return lhs + rhs


class IterativePipeline(Pipeline):
    def __init__(self):
        super(IterativePipeline, self).__init__()
        self.combinator = Combinator()

    def run(self, values):
        value = values[0]
        result = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        for i in range(2):
            value = self.combinator.combine(result, value[i])
            result = super(IterativePipeline, self).run(value)
        return result


def run_pipeline():

    p = IterativePipeline()
    return p.run(input_values)

__author__ = 'cclamb'

import numpy as np
import cPickle as cp

def unpickle():
    pickle_file = open('data/stdpu-state.pkl', 'rb')
    return cp.load(pickle_file)


def pickle():
    router_values = [
        [
            (1, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ])),
            (2, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ])),
            (3, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ]))],
        [
            (1, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ])),
            (2, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ])),
            (3, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ]))
        ],
        [
            (1, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ])),
            (2, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ])),
            (3, np.array([
                    [[1, 0, 0],
                     [0, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ]))],
        ]
    input_values = [
        np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1]
        ]),
        np.array([
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 1]
        ])
    ]
    pickle_file = file('data/stdpu-state.pkl', 'wb')
    cp.dump((router_values, input_values), pickle_file)
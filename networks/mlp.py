__author__ = 'cclamb'

import theano
import theano.tensor as T


class Layer(object):

    def __init__(self, w_init, b_init, activation):
        """
        A layer of a neural network, computes s(wx+b) where s is a nonlinearity and x is the input vector.

        :param w_init: np.ndarray, shape=(n_output, n_input)
            Values to for weight matrix initialization.
        :param b_init: np.ndarray, shape=(n_output,)
            Values for bias vector initialization.
        :param activation: theano.tensor.elemwise.Elemwise
            Activation function for the layer output.
        """

        n_output, n_input = w_init.shape
        assert b_init.shape == (n_output,)

        self.weights = theano.shared(value=w_init.astype(theano.config.floatX),
                                     name='w',
                                     borrow=True)
        self.bias = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                                  name='b',
                                  borrow=True,
                                  broadcastable=(False, True))

        self.activation = activation
        self.params = [self.weights, self.bias]

    def output(self, x):
        """
        Compute this layer's output given as an input.

        :param x: theano.tensor.var.TensorVariable
            Theano symbolic variable for layer input.
        :return: output: theano.tensor.var.TensorVariable
            Mixed, biased, and activated x
        """
        lin_output = T.dot(self.weights, x) + self.bias
        return lin_output if self.activation is None else self.activation(lin_output)


class MultiLayerPerceptron(object):

    def __init__(self, w_init, b_init, activations):
        """
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :param w_init: list of np.ndarray, len=N
            Values to initialize the weight matrix in each layer to.
            The layer sizes will be inferred from the shape of each matrix in W_init
        :param b_init: list of np.ndarray, len=N
            Values to initialize the bias vector in each layer to
        :param activations: list of theano.tensor.elemwise.Elemwise, len=N
            Activation function for layer output for each layer
        """
        assert len(w_init) == len(b_init) == len(activations)

        self.layers = []
        for w, b, activation in zip(w_init, b_init, activations):
            self.layers.append(Layer(w, b, activation))

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        """
        Compute the MLP's output given an input

        :param x: theano.tensor.var.TensorVariable
            Theano symbolic variable for network input
        :return: theano.tensor.var.TensorVariable
            x passed through the MLP
        """
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        """
        Compute the squared euclidean error of the network output against the "true" output y

        :param x: theano.tensor.var.TensorVariable
            Theano symbolic variable for network input
        :param y: theano.tensor.var.TensorVariable
            Theano symbolic variable for desired network output
        :return: theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        """
        return T.sum((self.output(x) - y)**2)


def gradient_updates_momentum(cost, params, rate, momentum):
    """
    Compute updates for gradient descent with momentum

    :param cost: theano.tensor.var.TensorVariable
        Theano cost function to minimize
    :param params: list of theano.tensor.var.TensorVariable
        Parameters to compute gradient against
    :param rate: float
        Gradient descent learning rate
    :param momentum: float
        Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
    :return: list
        List of updates, one for each parameter
    """
    assert 0 <= momentum < 1
    updates = []
    for param in params:
        update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - rate * update))
        updates.append((update, momentum * update + (1. - momentum) * T.grad(cost, param)))
    return updates



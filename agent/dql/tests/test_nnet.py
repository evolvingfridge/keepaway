import numpy as np
import unittest
from mock import patch

import theano
import theano.tensor as T

from ..nnet import (
    Layer,
    RectifiedLayer,
    NeuralNet,
)

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'


class TestLayer(unittest.TestCase):
    def test_layer(self):
        x = T.fmatrix('x')
        # minibatch_size = 6
        # nodes in prev layer = 2
        # nodes in layer = 4
        layer = Layer(x, 2, 4)
        f = theano.function(inputs=[x], outputs=[layer.output])
        # number of cols == numbers of nodes in prev layer
        minibatch = np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [7, 8],
        ])
        input = np.asarray(minibatch, dtype=theano.config.floatX)
        result = f(input)
        # weights: 4 columns (for each node in current layer), 2 rows (for each
        # node i previous layer)
        self.assertEqual(layer.weights.get_value().shape, (2, 4))
        # bias: 4 columns (one for each node in current layer)
        self.assertEqual(layer.bias.get_value().shape, (4,))
        # output: 3 rows (one for each sample in minibatches), 4 columns (one
        # for each node in current layer)
        self.assertEqual(result[0].shape, (6, 4))

    def test_rectify(self):
        value = np.array([[1, -3, 2, 0], [2, -4, 2, -1]])
        x = T.fmatrix('x')
        # minibatch_size = 3
        # nodes in prev layer = 2
        # nodes in layer = 4
        layer = RectifiedLayer(x, 2, 4)
        result = layer.rectify(value)
        np.testing.assert_array_equal(
            result,
            np.array([[1, 0, 2, 0], [2, 0, 2, 0]])
        )


class TestNeuralNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 5 nodes in first layer, 6 nodes in hidden layer, 3 nodes in output
        # layer
        cls.nnet = NeuralNet(n_inputs=2, architecture=[5, 6, 3])

    def test_layers(self):
        self.assertEqual(len(self.nnet.layers), 3)
        # weights and bias for each layer
        self.assertEqual(len(self.nnet.params), 6)
        # connections
        self.assertEqual(self.nnet.layers[1].input, self.nnet.layers[0].output)
        self.assertEqual(self.nnet.layers[2].input, self.nnet.layers[1].output)
        # sizes
        # for first layer every node will get one input
        self.assertEqual(self.nnet.layers[0].weights.get_value().shape, (2, 5))
        self.assertEqual(self.nnet.layers[1].weights.get_value().shape, (5, 6))
        self.assertEqual(self.nnet.layers[2].weights.get_value().shape, (6, 3))
        # output layer
        self.assertEqual(self.nnet.output_layer, self.nnet.layers[-1])

    def _get_minibatch(self):
        # minibatch size is 4
        minibatch = [
            # prestates
            np.array(
                [[2, 3], [3, 4], [5, 6], [7, 8]],
                dtype=theano.config.floatX
            ),
            # actions
            np.array([1, 2]),
            # rewards
            np.array([10, 20]),
            # poststates
            np.array(
                [[9, 10], [11, 12], [13, 14], [15, 16]],
                dtype=theano.config.floatX
            ),
            # is terminal
            np.array([False, False])
        ]
        return minibatch

    def test_predict(self):
        minibatch = self._get_minibatch()
        qvalues = self.nnet.predict(minibatch[0])[0]
        self.assertEqual(qvalues.shape, (4, 3))  # n_inputs * n_outputs

    def test_predict_best_action(self):
        action = self.nnet.predict_best_action(np.array([1, 2], dtype=theano.config.floatX))
        self.assertTrue(0 <= action <= 2)

    def test_train_minibatch(self):
        minibatch = self._get_minibatch()
        cost = self.nnet.train_minibatch(minibatch)
        self.assertEqual(type(cost.tolist()), float)

if __name__ == '__main__':
    unittest.main()

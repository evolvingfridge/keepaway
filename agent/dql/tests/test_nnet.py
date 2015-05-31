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
        weights = np.asarray(np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ]), dtype=theano.config.floatX)
        bias = np.asarray(np.array(
            [100, 200, 300, 400]
        ), dtype=theano.config.floatX)
        layer.weights.set_value(weights)
        layer.bias.set_value(bias)
        result = f(input)
        # weights: 4 columns (for each node in current layer), 2 rows (for each
        # node i previous layer)
        self.assertEqual(layer.weights.get_value().shape, (2, 4))
        # bias: 4 columns (one for each node in current layer)
        self.assertEqual(layer.bias.get_value().shape, (4,))
        # output: 3 rows (one for each sample in minibatches), 4 columns (one
        # for each node in current layer)
        self.assertEqual(result[0].shape, (6, 4))
        np.testing.assert_array_equal(result[0][0], np.array(
            [210, 340, 470, 600],  # (1*10+2*50, 1*20+2*60, 1*30+2*70, 1*40+2*80) + 100 (bias)
        ))

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
            np.array([1, 2, 0, 1]),
            # rewards
            np.array([10, 20, 30, 40]),
            # poststates
            np.array(
                [[9, 10], [11, 12], [13, 14], [15, 16]],
                dtype=theano.config.floatX
            ),
            # is terminal
            np.array([False, True, False, False])
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


class TestSimpleNeuralNet(unittest.TestCase):
    def setUp(self):
        # 5 nodes in first layer, 6 nodes in hidden layer, 3 nodes in output
        # layer
        weights = map(lambda x: np.asarray(x, dtype=theano.config.floatX), [
            np.array([
                [10, 20, 30],  # from first input to all nodes
                [40, 50, 60],  # from second input to all nodes
            ]),  # from input to layer 1
            [
                [100, 200],  # from node 1 in layer 1
                [300, 400],  # from node 1 in layer 1
                [500, 600],  # from node 1 to
            ],  # from layer 1 to layer 2
            [
                [-1, 2],
                [3, 4],
            ],  # from layer 2 to layer 3 (output layer)
        ])
        bias = map(lambda x: np.asarray(x, dtype=theano.config.floatX), [
            [1, 2, 3],
            [40, 50],
            [60, 70]
        ])
        self.nnet = NeuralNet(
            n_inputs=2,
            architecture=[3, 2, 2],
            l1_weight=1,
            l2_weight=2,
            _weights_values=weights,
            _bias_values=bias,
        )

    def test_output(self):
        # layer 1:
            # node1: 2 * 10 + 3 * 40 + 1 = 141
            # node2: 2 * 20 + 3 * 50 + 2 = 192
            # node2: 2 * 30 + 3 * 60 + 3 = 243

        # layer 2:
            # node1: 141 * 100 + 192 * 300 + 243 * 500 + 40 = 193 240
            # node2: 141 * 200 + 192 * 400 + 243 * 600 + 50 = 250 850

        # layer3:
            # node1: 193240 * -1 + 250850 * 3 + 60 = 559 370
            # node1: 193240 * 2 + 250850 * 4 + 70 = 1 389 950
        result = self.nnet.predict([[2, 3]])
        np.testing.assert_array_equal(result[0][0], np.asarray([559370, 1389950], dtype=theano.config.floatX))

    def test_error(self):
        cost = self.nnet.train([[2, 3]], [[559300, 1390000]])
        # cost = l1_weight * l1 + l2_weight * l2 + output_layer_error
        # l1 = sum of weights = 210 + 2100 + 10 = 2320
        # l2 = sum of weights squares = 9100 + 910 000 + 30 = 919 130
        # output layer error = (70 + 50) / 2 = 60
        self.assertEqual(cost[0], 2320 * 1 + 919130 * 2 + 60)

if __name__ == '__main__':
    unittest.main()

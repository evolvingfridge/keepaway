# neural net based on http://deeplearning.net/tutorial/code/mlp.py
import logging

import theano
import numpy as np
import theano.tensor as T

theano.config.openmp = False  # they say that using openmp becomes efficient only with "very large scale convolution"
theano.config.floatX = 'float32'

logger = logging.getLogger('keepaway')


class Layer(object):
    def __init__(self, input, n_inputs, n_nodes, activation=None, weights=None, bias=None):
        """
        Initialize a neural network layer.

        :param input: inputs from previous layer of shape (batch_size, n_inputs)
        :type input: theano.tensor.dmatrix

        :param n_inputs: number of inputs to single neuron (number of nodes in
            previous layer)
        :type n_inputs: int

        :param n_nodes: number of nodes in the layer. Also the size of output
        :type n_nodes: int
        """
        self.input = input
        # from RMSProp
        weight_bound = np.sqrt(6. / (n_inputs + n_nodes))
        # weights matrix of size n_inputs * n_nodes
        # each column (total: n_nodes) represents the weights from the input
        # units to the i-th unit
        if weights is not None:
            self.weights_values = weights
        else:
            self.weights_values = np.asarray(np.random.uniform(
                high=weight_bound,
                low=-weight_bound,
                size=(n_inputs, n_nodes)
            ), dtype=theano.config.floatX)
        self.weights = theano.shared(
            value=self.weights_values,
            name='weights',
            borrow=True,  # use "reference", not copy (http://deeplearning.net/software/theano/tutorial/aliasing.html#borrowing-when-creating-shared-variables)
        )
        # bias term
        if bias is not None:
            self.bias_values = bias
        else:
            self.bias_values = np.zeros((n_nodes,), dtype=theano.config.floatX)
        self.bias = theano.shared(value=self.bias_values, name='bias', borrow=True)
        # all the variables that can change during learning
        self.params = [self.weights, self.bias]

        # output
        # dot - returns inner product of params
        # For 2-D arrays it is equivalent to matrix multiplication
        # http://www.deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.dot
        # http://en.wikipedia.org/wiki/Matrix_multiplication#Inner_product
        self.output = T.dot(self.input, self.weights) + self.bias

        if activation:
            self.output = activation(self.output)

    def errors(self, y):
        """
        Return the error made in predicting the output value
        :param y: vector that gives for each node the value we wished to obtain
        :type y: theano.tensor.TensorType
        """
        return T.mean(T.abs_(self.output - y))


class RectifiedLayer(Layer):
    """
    Layer with minimum values of output equal to 0.
    """
    def __init__(self, *args, **kwargs):
        super(RectifiedLayer, self).__init__(*args, **kwargs)
        self.threshold = 0.
        # Output is rectified
        # self.output = self.rectify(self.output)
        self.output = T.maximum(self.output, self.threshold)

    def rectify(self, result):
        """
        Return max of result and threshold.
        """
        self._lin_output = result
        above_threshold = result > self.threshold
        return above_threshold * (result - self.threshold)


class OutputLayer(Layer):
    pass


class NeuralNet(object):
    discount_factor = 0.99
    learning_rate = 0.001
    l1_weight = 0.0
    l2_weight = 0.0

    train_batch = True

    # RMSprop params
    use_rmsprop = True
    rmsprop_rho = 0.9
    rmsprop_epsilon = 1e-6

    # type of x (input) variable
    x_type = T.fmatrix
    y_type = T.fmatrix

    # set weights and bias through __init__ kwargs if you want to have
    # it predefined
    _weights_values = None
    _bias_values = None

    def __init__(
        self,
        n_inputs,  # number of inputs to neural net
        architecture,
        **kwargs
    ):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
        # create theano variables corresponding to input_batch (x) and output
        # of the network (y)
        x = self.x_type('x')
        y = self.y_type('y')

        self.layers = []
        self.params = []
        # self.params_raw = []
        prev_layer = None
        prev_layer_size = None

        # create layers based on architecture
        for i, layer_size in enumerate(architecture):
            # params: inputs, number of inputs for single neuron, number of neurons
            layer_type = OutputLayer if i == len(architecture) - 1 else RectifiedLayer
            layer_kwargs = {}
            if self._weights_values is not None:
                layer_kwargs['weights'] = self._weights_values[i]
            if self._bias_values is not None:
                layer_kwargs['bias'] = self._bias_values[i]
            layer = layer_type(
                prev_layer.output if prev_layer else x,
                prev_layer_size or n_inputs,
                layer_size,
                **layer_kwargs
            )
            self.layers.append(layer)
            prev_layer = layer
            prev_layer_size = layer_size
            self.params.extend(layer.params)
            # self.params_raw.extend((layer.weights_values, layer.bias_values))

        self.output_layer = layer

        # define regularization terms, for some reason we only take in count
        # the weights, not biases) linear regularization term, useful for
        # having many weights zero
        # self.l1 = sum([abs(l.weights).sum() for l in self.layers])

        # square regularization term, useful for forcing small weights
        # self.l2_sqr = sum([(l.weights ** 2).sum() for l in self.layers])

        # define the cost function
        # self.cost = (
        #     self.l1_weight * self.l1 +
        #     self.l2_weight * self.l2_sqr +
        #     self.output_layer.errors(y)
        # )
        self.cost = self.output_layer.errors(y)

        updates = self._get_updates()

        # we need another set of theano variables (other than x and y) to use
        # in train and predict functions
        temp_x = self.x_type('temp_x')
        temp_y = self.y_type('temp_y')

        # define the training operation as applying the updates calculated
        # given temp_x and temp_y
        self.train = theano.function(
            inputs=[temp_x, temp_y],
            outputs=[self.cost],
            updates=updates,
            givens={  # specific substitutions to make in the computation graph
               x: temp_x,  # temp_x will replace x
               y: temp_y,  # temp_y will replace y
            },
            allow_input_downcast=True,
        )

        # output of network should be Q-value for an action, which is predicted
        # sum of future rewards
        self.predict = theano.function(
            inputs=[temp_x],
            outputs=[self.output_layer.output],
            givens={
                x: temp_x,
            },
            allow_input_downcast=True,
        )

    @property
    def params_raw(self):
        p = []
        for layer in self.layers:
            p.extend((layer.weights.get_value(), layer.bias.get_value()))
        return p

    def _get_updates(self):
        """
        Calculate params updates using RMSProp

        Based on:
        http://nbviewer.ipython.org/github/udibr/Theano-Tutorials/blob/master/notebooks/4_modern_net.ipynb
        https://www.youtube.com/watch?v=O3sxAc4hxZU
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

        In short, Hinton suggests "the magnitude of the gradient can be very
        different for different weights and can change during learning. This
        makes it hard to choose a global learning rate." RMSProp solves this
        problem by "dividing the learning rate for a weight by a running
        average of the magnitudes of recent gradients for that weight."
        """
        # define gradient calculation
        self.grads = grads = T.grad(self.cost, self.params)

        logger.debug('gradient: {}'.format(grads))
        # Define how much we need to change the parameter values
        # actual RMSProp
        updates = []
        for param_i, gparam_i in zip(self.params, grads):
            if self.use_rmsprop:
                # acc is allocated for each parameter (param_i) with 0 values with the shape of p
                acc = theano.shared(param_i.get_value() * 0.)
                acc_new = self.rmsprop_rho * acc + (1 - self.rmsprop_rho) * gparam_i ** 2
                gradient_scaling = T.sqrt(acc_new + self.rmsprop_epsilon)
                gparam_i = gparam_i / gradient_scaling
                updates.append((acc, acc_new))
            updates.append((param_i, param_i - self.learning_rate * gparam_i))
        logger.debug('Updates: {}'.format(updates))
        return updates

    def train_minibatch(self, minibatch):
        """
        Train minibatch using Q-learning
        """
        # logger.debug('Training minibatch: {}'.format(minibatch))
        prestates, actions, rewards, poststates, terminals = minibatch

        # predict Q-values for prestates, so we can keep Q-values for other
        # actions unchanged
        qvalues = self.predict(prestates)[0]
        logger.debug('Predicted Q-values: {}'.format(qvalues))
        # predict Q-values for poststates
        post_qvalues = self.predict(poststates)[0]
        logger.debug('Predicted post-Q-values: {}'.format(post_qvalues))
        # take maximum Q-value of all actions (future reward for terminal state
        # is 0)
        max_qvalues = np.max(post_qvalues, axis=1) * (1 - terminals)
        logger.debug('Max Q-values: {}'.format(max_qvalues))
        # update the Q-values for the actions we actually performed
        # Q*(s, a) = r + (1 - terminal) * discount_factor * max_a Q(s', a)
        for i, action in enumerate(actions):
            qvalues[i][action] = rewards[i] + self.discount_factor * max_qvalues[i]
        logger.debug('Updated Q-values (Q-learning): {}'.format(qvalues))

        if self.train_batch:
            cost = self.train(prestates, qvalues)[0]
        else:
            cost = 0
            for prestate, qval in zip(prestates, qvalues):
                cost += self.train([prestate], [qval])[0]
        logger.debug('Current error: {}'.format(cost))
        return cost

    def predict_best_action(self, state):
        """
        Returns the action with the highest Q-value
        """
        q_values = self.predict([state])[0]
        logger.debug('Q values: {}'.format(q_values))
        return np.argmax(q_values), np.max(q_values)

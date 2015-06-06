import logging
import numpy as np
import theano
from theano import tensor as T

logger = logging.getLogger('keepaway')


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


def model(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx


class NeuralNetSimple(object):
    def __init__(self, *args, **kwargs):
        X = T.fmatrix()
        Y = T.fmatrix()

        w_h = init_weights((13, 30))
        w_o = init_weights((30, 3))

        py_x = model(X, w_h, w_o)
        # y_x = [
        #     T.argmax(py_x, axis=1),
        #     T.max(py_x, axis=1),
        # ]

        cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        params = [w_h, w_o]
        updates = sgd(cost, params)

        self.train = theano.function(inputs=[X, Y], outputs=[cost], updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[X], outputs=[py_x], allow_input_downcast=True)

    @property
    def params_raw(self):
        return []

    def train_minibatch(self, minibatch):
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
        for i, action in enumerate(actions):
            qvalues[i][action] = rewards[i] + max_qvalues[i]
        logger.debug('Updated Q-values (Q-learning): {}'.format(qvalues))
        cost = self.train(prestates, qvalues)[0]
        logger.debug('Current error: {}'.format(cost))
        return cost

    def predict_best_action(self, state):
        """
        Returns the action with the highest Q-value
        """
        q_values = self.predict([state])[0]
        logger.debug('Q values: {}'.format(q_values))
        return np.argmax(q_values), np.max(q_values)

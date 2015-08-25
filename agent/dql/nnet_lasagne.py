"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""
import logging

import lasagne
import numpy as np
import theano
import theano.tensor as T
# from updates import deepmind_rmsprop

theano.config.openmp = False  # they say that using openmp becomes efficient only with "very large scale convolution"
theano.config.floatX = 'float64'

logger = logging.getLogger('keepaway')


class NeuralNetLasagne(object):
    """
    Deep Q-learning network using Lasagne.
    """
    number_of_actions = 3  # num_actions
    state_size = 13

    discount_factor = 1.0  # discount
    minibatch_size = 8  # batch_size
    recent_states_to_network = 1

    initial_epsilon_greedy = 1  # every action is random action
    final_epsilon_greedy = 0.0  # every action is not random
    exploration_time = float(10**4)  # number of episodes over which epsilon factor is linearly annealed to it's final value
    start_learn_after = 5 * 10**2

    network_architecture = [13, 30, 100, 30, 3]

    start_learning_rate = 0.00005
    final_learning_rate = 0.00005
    learning_rate_change_episodes = 5 * 10**3

    use_rmsprop = True
    rmsprop_rho = 0.9  # rho
    rmsprop_epsilon = 1e-6

    clip_delta = 0

    error_func = 'sum'
    update_rule = 'rmsprop'
    _swap_networks_every = 0  # freeze interval

    rng = np.random.RandomState()

    def __init__(self, n_inputs, architecture, **kwargs):

        for kw_name, kw_val in kwargs.items():
            try:
                setattr(self, kw_name, kw_val)
            except AttributeError:
                pass

        self.frames_played = 0
        self.episodes_played = 0

        self.output_layer = self._build_network()
        if self._swap_networks_every > 0:
            self.next_output_layer = self._build_network()
            self._swap_networks()

        states = T.matrix('states')
        next_states = T.matrix('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        self.states_dimensions = (
            self.minibatch_size, self.state_size
        )
        self.states_shared = theano.shared(np.zeros(
            self.states_dimensions,
            dtype=theano.config.floatX
        ))

        self.next_states_shared = theano.shared(np.zeros(
            self.states_dimensions,
            dtype=theano.config.floatX
        ))

        self.rewards_shared = theano.shared(
            np.zeros((self.minibatch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True)
        )

        self.actions_shared = theano.shared(
            np.zeros((self.minibatch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((self.minibatch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        q_values = lasagne.layers.get_output(self.output_layer, states)

        if self._swap_networks_every > 0:
            next_q_values = lasagne.layers.get_output(
                self.next_output_layer, next_states
            )
        else:
            next_q_values = lasagne.layers.get_output(
                self.output_layer, next_states)
            next_q_values = theano.gradient.disconnected_grad(next_q_values)

        target = (
            rewards +
            (T.ones_like(terminals) - terminals) *
            self.discount_factor * T.max(next_q_values, axis=1, keepdims=True)
        )
        diff = target - q_values[
            T.arange(self.minibatch_size),
            actions.reshape((-1,))
        ].reshape((-1, 1))

        if self.clip_delta > 0:
            diff = diff.clip(-self.clip_delta, self.clip_delta)

        if self.error_func == 'sum':
            loss = T.sum(diff ** 2)
        elif self.error_func == 'mean':
            loss = T.mean(diff ** 2)

        params = lasagne.layers.helper.get_all_params(self.output_layer)
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }

        self.learning_rate = theano.shared(np.cast['float64'](0))

        if self.update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(
                loss, params, self.learning_rate, self.rmsprop_rho,
                self.rmsprop_epsilon
            )
        elif self.update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.learning_rate)
        else:
            raise ValueError("Unrecognized update: {}".format(self.update_rule))

        self._train = theano.function(
            [], [loss, q_values], updates=updates, givens=givens
        )
        self._q_values = theano.function(
            [], q_values, givens={states: self.states_shared}
        )
        print(self)

    def __str__(self):
        result = ['Lasagne NNET config: ']
        for v in [
            'discount_factor', 'train_batch', 'use_rmsprop', 'rmsprop_rho',
            'rmsprop_epsilon'
        ]:
            result.append('{}: {}'.format(v, getattr(self, v)))
        return '\n'.join(result)

    def train_minibatch(self, minibatch, learning_rate):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        self.learning_rate.set_value(learning_rate)
        states, actions, rewards, next_states, terminals = minibatch
        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        if (
            self._swap_networks_every > 0 and
            self.frames_played % self._swap_networks_every == 0
        ):
            self._swap_networks()
        loss, _ = self._train()
        self.frames_played += 1
        return np.sqrt(loss)

    def _get_state_qvalues(self, state):
        states = np.zeros(self.states_dimensions, dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        return self._q_values()[0]

    def predict_best_action(self, state):
        q_values = self._get_state_qvalues(state)
        return np.argmax(q_values), np.max(q_values)

    def _swap_networks(self):
        all_params = lasagne.layers.helper.get_all_param_values(
            self.output_layer
        )
        lasagne.layers.helper.set_all_param_values(
            self.next_output_layer, all_params
        )

    def _build_network(self):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        l_in = lasagne.layers.InputLayer(
            shape=(
                self.minibatch_size,
                self.recent_states_to_network,
                self.state_size
            )
        )
        l_prev = l_in
        for layer_size in self.network_architecture[1:-1]:
            l_prev = lasagne.layers.DenseLayer(
                l_prev,
                num_units=layer_size,
                nonlinearity=lasagne.nonlinearities.rectify
            )

        output_layer = lasagne.layers.DenseLayer(
            l_prev,
            num_units=self.number_of_actions,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)  # None?
        )

        return output_layer

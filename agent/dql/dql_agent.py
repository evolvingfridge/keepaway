import random

from .states_memory import TransitionTable
from .nnet import NeuralNet


class DQLAgent(object):
    # number of most recent transitions (s, a, r, s') in history
    transitions_history_size = 10**6
    # minibatch size
    minibatch_size = 2
    # number of most recent states that are given as input to network
    recent_states_to_network = 1
    # discount factor
    discount_factor = 0.99
    # learning rare
    learning_rate = 0.00025
    # epsilon-greedy factors
    initial_epsilon_greedy = 1  # every action is random action
    final_epsilon_greedy = 0.1  # one for ten actions is random
    exploration_time = float(10**6)  # number of frames over which epsilon factor is linearly annealed to it's final value
    # start learn after X frames
    start_learn_after = 5*(10**4)
    # network architecture (first layer is number of inputs, last is number of actions)
    network_architecture = [13, 30, 30, 3]
    # possible number of actions
    number_of_actions = 3
    # state size
    state_size = 13
    # if training mode
    train = False

    @property
    def epsilon(self):
        return max(
            self.initial_epsilon_greedy - (self.frames_played - self.start_learn_after) / self.exploration_time,
            self.final_epsilon_greedy
        )

    def __init__(self, **kwargs):
        for kw_name, kw_val in kwargs.iteritems():
            setattr(self, kw_name, kw_val)
        self.memory = TransitionTable(
            self.transitions_history_size,
            state_size=self.state_size,
            full_state_samples_count=self.recent_states_to_network,
        )
        self.nnet = NeuralNet(
            n_inputs=self.state_size,
            architecture=self.network_architecture,
            discount_factor=self.discount_factor,
            learning_rate=self.learning_rate
        )
        self.frames_played = 0
        self.scores = []
        self._init_new_game()

    def _init_new_game(self):
        self.last_state = None
        self.last_action = None
        self.current_game_total_reward = 0
        # self.memory.add(np.zeros((self.state_size,)), 0, 0, False)

    def _train_minibatch(self):
        minibatch = self.memory.get_minibatch(self.minibatch_size)
        self.nnet.train_minibatch(minibatch)

    def _remember_in_memory(self, reward, is_terminal=False):
        """
        Save in memory last state, last action done, reward and information
        if after making last action there was a terminal state.
        """
        self.memory.add(self.last_state, self.last_action, reward, is_terminal)
        self.current_game_total_reward += reward

    def _get_next_action(self):
        """
        Return next action to be done.
        """
        current_state = self.memory.get_last_full_state()
        if random.uniform(0, 1) < self.epsilon or self.frames_played < self.start_learn_after:
            action = random.choice(range(self.number_of_actions))
        else:
            action = self.nnet.predict(current_state)
        self.last_action = action
        self.frames_played += 1
        return action

    # ==================================

    def start_episode(self, *args, **kwargs):
        # super(DQLAgent, self).start_episode(*args, **kwargs)
        self._init_new_game()
        return self.step(*args, **kwargs)

    def step(self, reward, current_state, *args, **kwargs):
        # super(DQLAgent, self).step(reward, current_state, *args, **kwargs)
        if self.last_state is not None:
            self._remember_in_memory(reward)
        if self.train and self.frames_played > self.start_learn_after:
            self._train_minibatch()
        self.last_action = self._get_next_action()
        self.last_state = current_state
        return self.last_action

    def end_episode(self, reward, *args, **kwargs):
        # super(DQLAgent, self).game_over(reward, *args, **kwargs)
        if self.last_state is not None:
            self._remember_in_memory(reward, True)
        self.scores.append(self.current_game_total_reward)

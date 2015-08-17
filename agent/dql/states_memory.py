import numpy as np
import random


class TransitionTable(object):
    # number of consecutive states which create one full-step (ex. for input to neural net)
    full_state_samples_count = 1

    def __init__(self, n, state_size, **kwargs):
        """
        Initialize memory structure
        :param n: the number of game steps
        :param state_size: size of single game state
        """
        self.size = n
        self.state_size = state_size
        self.states = np.empty((n, state_size), dtype=np.float32)
        self.actions = np.empty((n,), dtype=np.uint8)
        self.rewards = np.empty((n,), dtype=np.float32)
        self.is_terminal = np.empty((n,), dtype=bool)
        self.recently_saved_index = -1
        self.entries_count = 0

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._terminal_state = np.zeros((state_size * self.full_state_samples_count,), dtype=np.float32)

    # @property
    # def next_save_index(self):
    #     self.recently_saved_index = (self.recently_saved_index + 1) % self.size
    #     return self.recently_saved_index

    def add(self, state, action, reward, is_terminal=False):
        """
        Store (state, action, reward, is_terminal) information in memory
        (what was the state, what action was done, what was the reward after
        applying such action and if state after making such action was terminal
        state).
        """
        self.recently_saved_index = (self.recently_saved_index + 1) % self.size
        i = self.recently_saved_index
        self.actions[i] = action
        self.rewards[i] = reward
        self.states[i] = state
        self.is_terminal[i] = is_terminal
        self.entries_count += 1

    def get_minibatch(self, samples):
        """
        Return `samples` (state, action, reward, state') tuples
        :param samples: samples of the minibatch
        """
        states_size = self.state_size * self.full_state_samples_count
        prestates = np.empty((samples, states_size), dtype = np.float32)
        actions = np.empty((samples), dtype=np.float32)
        rewards = np.empty((samples), dtype=np.float32)
        poststates = np.empty((samples, states_size), dtype = np.float32)
        terminals = np.empty((samples), dtype=bool)

        # Pick random `size` states
        j = 0
        while j < samples:
            state, action, reward, poststate, is_terminal = self._get_random_sample()

            prestates[j, :] = state  # copy state to index j
            actions[j] = action
            rewards[j] = reward
            poststates[j, :] = poststate
            terminals[j] = is_terminal
            j += 1

        return [prestates, actions, rewards, poststates, terminals]

    def _get_state(self, index):
        """
        Join states into full state
        """
        states = []
        for i in range(index, index + self.full_state_samples_count):
            i = i % self.size
            states.append(np.copy(self.states[i]))  # append copy
            # clear previous states and start from new non-terminal
            if i != (index + self.full_state_samples_count - 1) % self.size and self.is_terminal[i]:
                for s in states:
                    s.fill(0)
        return np.ravel(states)

    def _is_state_valid(self, prestate_start, prestate_end):
        return True  # TODO

    def _get_random_sample(self):
        # assert(self.entries_count >= 1)
        max_index = np.min([self.entries_count, self.size]) - 1
        while True:
            # single sample will occupy elements from i to i2 (inclusive)
            i = random.randint(0, max_index)
            i2_1 = (i + self.full_state_samples_count - 1)
            i2 = i2_1 % self.size
            # not self.is_terminal[i2],  # last element isn't terminal
            # check if states from i to i2+1 (inclusive) are continuous
            if self.recently_saved_index not in set(
                map(lambda x: x % self.size, range(i, i2_1 + 1))
            ):
                break
            # # check if full state is continuous
            # if self.full_state_samples_count > 1 and not self._is_state_valid(i, i2_1):
            #     continue
            # break

        state = self._get_state(i)
        action = self.actions[i2]
        reward = self.rewards[i2]
        is_terminal = self.is_terminal[i2]
        if is_terminal:
            poststate = self._terminal_state
        else:
            poststate = self._get_state(i + 1)
        return state, action, reward, poststate, is_terminal

    def get_last_state(self):
        return self.states[self.recently_saved_index % self.size]

    def get_last_full_state(self, current_state):
        index = (self.recently_saved_index + self.size - self.full_state_samples_count) % self.size
        states = []
        for i in range(index, index + self.full_state_samples_count - 1):
            i = i % self.size
            states.append(np.copy(self.states[i]))  # append copy
            # clear previous states and start from new non-terminal
            if i != (index + self.full_state_samples_count - 1) % self.size and self.is_terminal[i]:
                for s in states:
                    s.fill(0)
        states.append(current_state)
        return np.ravel(states)

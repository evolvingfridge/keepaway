import numpy as np
import unittest
from itertools import cycle
from mock import patch

from ..states_memory import TransitionTable


class RandomMock(object):
    calls = 0
    values = [1, 2, 3]

    @staticmethod
    def random(*args):
        val = RandomMock.values[RandomMock.calls % len(RandomMock.values)]
        RandomMock.calls += 1
        return val


class TestMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.__state = np.array([1, 2, 3])
        cls.__action = 1
        cls.__reward = 2

    def setUp(self):
        self.mem = TransitionTable(5, 3)

    @property
    def _state(self):
        self.__state += 1
        return self.__state

    @property
    def _action(self):
        self.__action += 1
        return self.__action

    @property
    def _reward(self):
        self.__reward += 1
        return self.__reward

    def test_next_index(self):
        self.assertEqual(self.mem.next_save_index, 0)
        self.assertEqual(self.mem.next_save_index, 1)
        self.assertEqual(self.mem.next_save_index, 2)
        self.assertEqual(self.mem.next_save_index, 3)
        self.assertEqual(self.mem.next_save_index, 4)
        self.assertEqual(self.mem.next_save_index, 0)

    def _add_sample(self, state=None, action=None, reward=None, is_terminal=False):
        next_state = np.array(state if state is not None else self._state)
        action = action or self._action
        reward = reward or self._reward
        self.mem.add(next_state, action, reward, is_terminal)

    def test_add(self):
        self._add_sample(np.array([1, 2, 3]), 1, 2)
        np.testing.assert_array_equal(self.mem.states[0], np.array([1, 2, 3]))
        self.assertEqual(self.mem.actions[0], 1)
        self.assertEqual(self.mem.rewards[0], 2)
        self.assertEqual(self.mem.recently_saved_index, 0)

    @patch('dql.states_memory.random.randint')
    def test_get_random_sample(self, randint_mock):
        self.__state = np.array([1, 2, 3])
        self.__action = 10
        self.__reward = 20

        self._add_sample()
        self._add_sample()
        self._add_sample(is_terminal=True)
        self._add_sample()
        self._add_sample()

        RandomMock.calls = 0
        RandomMock.values = [2, 3, 1]
        randint_mock.side_effect = RandomMock.random

        sample = self.mem._get_random_sample()
        prestate, action, reward, poststate, terminal = sample
        # 2 - is terminal
        # 3 - ok
        np.testing.assert_array_equal(prestate, self.mem.states[3])
        np.testing.assert_array_equal(poststate, self.mem.states[4])
        self.assertEqual(action, self.mem.actions[3])
        self.assertEqual(reward, self.mem.rewards[3])
        self.assertEqual(terminal, self.mem.is_terminal[4])

    @patch('dql.states_memory.random.randint')
    def test_get_random_sample_size2(self, randint_mock):
        self.mem.full_state_samples_count = 2
        self.__state = np.array([1, 2, 3])
        self.__action = 10
        self.__reward = 20

        self._add_sample()
        self._add_sample()
        self._add_sample(is_terminal=True)
        self._add_sample()
        self._add_sample()

        RandomMock.calls = 0
        RandomMock.values = [0, 1, 2]
        randint_mock.side_effect = RandomMock.random

        sample = self.mem._get_random_sample()
        prestate, action, reward, poststate, terminal = sample
        np.testing.assert_array_equal(prestate, np.array([2, 3, 4, 3, 4, 5]))
        np.testing.assert_array_equal(poststate, np.array([3, 4, 5, 4, 5, 6]))
        self.assertEqual(action, self.mem.actions[1])
        self.assertEqual(reward, self.mem.rewards[1])
        self.assertEqual(terminal, self.mem.is_terminal[2])

    @patch('dql.states_memory.random.randint')
    def test_get_random_sample_intersection(self, randint_mock):
        self.mem.full_state_samples_count = 2
        self.__state = np.array([1, 2, 3])
        self.__action = 10
        self.__reward = 20

        self._add_sample()
        self._add_sample()
        self._add_sample(is_terminal=True)
        self._add_sample()
        self._add_sample()

        RandomMock.calls = 0
        RandomMock.values = [4, 0]
        randint_mock.side_effect = RandomMock.random

        sample = self.mem._get_random_sample()
        prestate, action, reward, poststate, terminal = sample
        np.testing.assert_array_equal(prestate, np.array([2, 3, 4, 3, 4, 5]))
        np.testing.assert_array_equal(poststate, np.array([3, 4, 5, 4, 5, 6]))
        self.assertEqual(action, self.mem.actions[1])
        self.assertEqual(reward, self.mem.rewards[1])
        self.assertEqual(terminal, self.mem.is_terminal[2])

    def test_get_minibatch(self):
        self.mem.full_state_samples_count = 2
        self.__state = np.array([1, 2, 3])
        self.__action = 10
        self.__reward = 20

        self._add_sample()
        self._add_sample()
        self._add_sample(is_terminal=True)
        self._add_sample()
        self._add_sample()
        minibatch = self.mem.get_minibatch(2)
        prestates, actions, rewards, poststates, terminals = minibatch
        self.assertEqual(prestates.shape, (2, 6))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(poststates.shape, (2, 6))


class TestMemory2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.__state = np.array([1, 2, 3])
        cls.__action = 1
        cls.__reward = 2

    def setUp(self):
        self.mem = TransitionTable(2, 3)

    def _add_samples(self, count, terminals=None):
        terminals = terminals or set()
        for i in range(1, count + 1):
            self.mem.add(i, i, i, i in terminals)

    def test_get_random_sample_even(self):
        self._add_samples(4)
        for r, rand_call_expected in [(cycle([0, 1]), 1), (cycle([1, 0]), 2)]:
            with patch('dql.states_memory.random.randint') as rand_mock:
                def rand(*args, **kwargs):
                    return r.next()

                rand_mock.side_effect = rand

                prestate, action, reward, poststate, terminal = self.mem._get_random_sample()
                np.testing.assert_array_equal(prestate, [3, 3, 3])
                np.testing.assert_array_equal(poststate, [4, 4, 4])
                np.testing.assert_array_equal(action, 3)
                np.testing.assert_array_equal(reward, 3)
                self.assertEqual(rand_mock.call_count, rand_call_expected)

    def test_get_random_sample_odd(self):
        self._add_samples(5)
        for r, rand_call_expected in [(cycle([0, 1]), 2), (cycle([1, 0]), 1)]:
            with patch('dql.states_memory.random.randint') as rand_mock:
                def rand(*args, **kwargs):
                    return r.next()

                rand_mock.side_effect = rand

                prestate, action, reward, poststate, terminal = self.mem._get_random_sample()
                np.testing.assert_array_equal(prestate, [4, 4, 4])
                np.testing.assert_array_equal(poststate, [5, 5, 5])
                np.testing.assert_array_equal(action, 4)
                np.testing.assert_array_equal(reward, 4)
                self.assertEqual(rand_mock.call_count, rand_call_expected)


if __name__ == '__main__':
    unittest.main()

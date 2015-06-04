import numpy as np
import unittest
from mock import patch

import theano

from ..dql_agent import DQLAgent

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'


class TestDQLAgent(unittest.TestCase):
    # def setUp(self):
    #     self.agent = DQLAgent(state_size=3)

    # def test_game(self):
    #     self.agent.start_episode()
    #     self.agent.step(0, [1, 2, 3])
    #     self.agent.step(1, [2, 3, 4])
    #     self.agent.step(2, [3, 4, 5])
    #     self.agent.game_over(3)

    #     self.assertEqual(self.agent.scores, [6])
    #     next_action = self.agent._get_next_action()
    #     self.assertLess(next_action, self.agent.number_of_actions)
    #     self.assertEqual(self.agent.frames_played, 4)
    #     self.assertEqual(self.agent.last_action, next_action)

    def _set_up_agent(
        self,
        **kwargs
    ):
        default_options = dict(
            transitions_history_size=1,
            minibatch_size=1,
            state_size=1,
            start_learn_after=100,
        )
        default_options.update(kwargs)
        self.agent = DQLAgent(**default_options)

    def _play_games(self):
        self.agent.start_episode(0, [1])
        self.agent.step(2, [2])
        self.agent.start_episode(3, [3])
        self.agent.step(3, [4])
        self.agent.end_episode(4)
        self.agent.end_episode(4)
        self.agent.end_episode(4)

        self.agent.start_episode(7, [5])
        self.agent.start_episode(9, [6])
        self.agent.step(10, [7])
        self.agent.start_episode(13, [8])
        self.agent.step(15, [9])
        self.agent.step(17, [10])
        self.agent.step(18, [11])
        self.agent.end_episode(20)
        self.agent.end_episode(20)
        self.agent.end_episode(20)

    def test_big_memory(self):
        self._set_up_agent(transitions_history_size=100)
        self._play_games()
        with patch('random.randint') as mock_random:
            self.n = -1

            def rand(*args, **kwargs):
                self.n += 1
                return self.n

            mock_random.side_effect = rand
            r = self.agent.memory.get_minibatch(11)
            # prestates
            np.testing.assert_array_equal(r[0], [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])
            # rewards
            np.testing.assert_array_equal(r[2], [2, 1, 0, 1, 2, 1, 3, 2, 2, 1, 2])
            # poststates
            np.testing.assert_array_equal(r[3], [[2], [3], [4], [0], [6], [7], [8], [9], [10], [11], [0]])
            # terminal
            np.testing.assert_array_equal(r[4], [False, False, False, True, False, False, False, False, False, False, True])

    def test_memory_1(self):
        self._set_up_agent(transitions_history_size=1)
        self._play_games()
        r = self.agent.memory.get_minibatch(1)
        # prestates
        np.testing.assert_array_equal(r[0], [[11]])
        # rewards
        np.testing.assert_array_equal(r[2], [2])
        # poststates
        np.testing.assert_array_equal(r[3], [[0]])
        # terminal
        np.testing.assert_array_equal(r[4], [True])

    def test_memory_2(self):
        self._set_up_agent(transitions_history_size=2)
        self._play_games()
        r = self.agent.memory.get_minibatch(1)
        import ipdb; ipdb.set_trace()
        # prestates
        np.testing.assert_array_equal(r[0], [[11]])
        # rewards
        np.testing.assert_array_equal(r[2], [2])
        # poststates
        np.testing.assert_array_equal(r[3], [[0]])
        # terminal
        np.testing.assert_array_equal(r[4], [True])

if __name__ == '__main__':
    unittest.main()

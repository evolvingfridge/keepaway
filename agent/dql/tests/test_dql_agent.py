import numpy as np
import unittest
from mock import patch

import theano
import theano.tensor as T

from ..dql_agent import DQLAgent

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'


class TestDQLAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DQLAgent(state_size=3)

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


if __name__ == '__main__':
    unittest.main()

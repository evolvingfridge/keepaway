import random

from .environment import PLAYER_X, EMPTY
from dql.dql_agent import DQLAgent


class Agent(object):
    def start_episode(self, *args, **kwargs):
        """
        Notification about beginning of the next episode
        """
        pass

    def step(self, reward, current_state, *args, **kwargs):
        """
        Return action for current state.

        :param reward: reward for previous move
        :type reward: int

        :param current_state: current state of the game
        :type current_state: np.array

        :returns: action index
        :rtype: int
        """
        pass

    def end_episode(self, reward, *args, **kwargs):
        """
        Notification about end of episode (single game over).

        :param reward: reward for previous move
        :type reward: int
        """
        pass


class TicTacToeAgent(Agent):
    def start_episode(self, marker=PLAYER_X):
        self.current_marker = marker


class RandomAgent(TicTacToeAgent):
    def next_move(self, board):
        empty_positions = []
        for row in range(len(board)):
            for col in range(len(board)):
                if board[row][col] == EMPTY:
                    empty_positions.append((row, col))
        rand = random.randint(0, len(empty_positions) - 1)
        return empty_positions[rand]


class TicTacToeDQLAgent(DQLAgent, TicTacToeAgent):
    pass

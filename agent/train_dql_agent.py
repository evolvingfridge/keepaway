from .agent import RandomAgent, TicTacToeDQLAgent
from .environment import TicTacToeEnvironment, PLAYER_O, PLAYER_X


frames = 10**2


def main():
    env = TicTacToeEnvironment()
    agent1 = TicTacToeDQLAgent()
    agent2 = TicTacToeDQLAgent()
    frames_played = 0
    while frames_played < frames:
        frames_played += env.play(agent1, agent2)
        env.print_board()
        print('=' * 20)

if __name__ == '__main__':
    main()

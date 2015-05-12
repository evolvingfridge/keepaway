from .agent import RandomAgent
from .environment import TicTacToeEnvironment, PLAYER_O, PLAYER_X


def main():
    env = TicTacToeEnvironment()
    random_agent = RandomAgent(PLAYER_X)
    random_agent2 = RandomAgent(PLAYER_O)
    for episode in range(10):
        env._generate_board()
        result = env.play(random_agent, random_agent2)
        print(result)
        env.print_board()
        print('=' * 20)

if __name__ == '__main__':
    main()

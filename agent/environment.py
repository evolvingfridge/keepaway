EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3

WINNER_REWARD = 10
LOOSED_REWARD = 0


class TicTacToeEnvironment(object):
    def __init__(self, board_size=3):
        self.board_size = board_size

    def _generate_board(self):
        self.board = [x[:] for x in [[EMPTY] * self.board_size] * self.board_size]
        return self.board

    def print_board(self):
        mapping = {
            EMPTY: '.',
            PLAYER_X: 'X',
            PLAYER_O: 'O',
        }
        row_h = '-' * (self.board_size * 3 - 2)
        for row in self.board:
            print(row_h)
            row_str = []
            for col in row:
                row_str.extend(['|', mapping[col]])
            row_str.append('|')
            print(''.join(map(str, row_str)))
        print(row_h)

    def is_over(self):
        def _is_over(values):
            v = set(values)
            return len(v) == 1 and v.pop()
        # check columns
        for col in range(self.board_size):
            values = [self.board[i][col] for i in range(self.board_size)]
            ov = _is_over(values)
            if ov:
                return ov

        # check rows
        for row in range(self.board_size):
            values = [self.board[row][i] for i in range(self.board_size)]
            ov = _is_over(values)
            if ov:
                return ov

        # check diagonals
        diag1 = [self.board[i][i] for i in range(self.board_size)]
        diag2 = [self.board[i][self.board_size - i - 1] for i in range(self.board_size)]
        ov = _is_over(diag1) or _is_over(diag2)
        if ov:
            return ov
        return all([v for y in self.board for v in y])

    def apply_move(self, player, move):
        if self.board[move[0]][move[1]] != EMPTY:
            raise Exception()
        assert player in (PLAYER_X, PLAYER_O)
        self.board[move[0]][move[1]] = player

    def _get_agent_reward(self, agent):
        is_over = self._is_over()
        if is_over:
            if agent.current_marker == is_over:
                return WINNER_REWARD
            else:
                return LOOSED_REWARD
        return 0

    def play(self, agent_x, agent_o):
        self._generate_board()
        frames = 0
        agents = [agent_x, agent_o]
        current_player = 0
        for agent, marker in zip(agents, (PLAYER_X, PLAYER_O)):
            agent.start_episode(marker=marker)

        while not self.is_over():
            current_agent = agents[current_player]
            next_move = current_agent.step(0, self.board)
            self.apply_move(current_player + 1, next_move)
            current_player = (current_player + 1) % 2
            frames += 1

        for agent in agents:
            reward = self._get_agent_reward(agent)
            agent.end_episode(reward)

        return frames

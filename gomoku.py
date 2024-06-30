# gomoku.py

class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = [[0] * board_size for _ in range(board_size)]
        self.current_player = 1

    def reset(self):
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1

    def is_valid_move(self, x, y):
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == 0

    def make_move(self, x, y):
        if self.is_valid_move(x, y):
            self.board[x][y] = self.current_player
            if self.check_win(x, y):
                return f'Player {self.current_player} wins!'
            self.current_player = 3 - self.current_player  # Switch player
            return 'Continue'
        return 'Invalid move'

    def check_win(self, x, y):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                if self._in_bounds(x + i * dx, y + i * dy) and self.board[x + i * dx][y + i * dy] == self.current_player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                if self._in_bounds(x - i * dx, y - i * dy) and self.board[x - i * dx][y - i * dy] == self.current_player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def _in_bounds(self, x, y):
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def get_empty_positions(self):
        return [(x, y) for x in range(self.board_size) for y in range(self.board_size) if self.board[x][y] == 0]

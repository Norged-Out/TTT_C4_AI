"""
Author: Priyansh Nayak
Description: Stores the Connect 4 game state and rules
"""


class Connect4:
    ROWS = 6
    COLS = 7

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # board is stored row-major, top to bottom
        self.board = [[" " for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.current_player = "X"
        self.winner = None
        self.last_move = None

    def available_moves(self) -> list[int]:
        # a column is playable if its top cell is still empty
        moves = []
        for col in range(self.COLS):
            if self.board[0][col] == " ":
                moves.append(col)
        return moves

    def make_move(self, col: int) -> bool:
        # reject moves after game ends
        if self.winner is not None:
            return False

        # reject invalid or full columns
        if col < 0 or col >= self.COLS or col not in self.available_moves():
            return False

        # token falls to the lowest empty row in the chosen column
        row = self.get_drop_row(col)
        if row is None:
            return False

        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        self.winner = self.check_winner(row, col)

        if self.winner is None:
            self.current_player = "O" if self.current_player == "X" else "X"

        return True

    def get_drop_row(self, col: int):
        # search upward from the bottom because tokens fall down
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][col] == " ":
                return row
        return None

    def count_in_direction(self, row: int, col: int, dr: int, dc: int) -> int:
        # count matching tokens starting just beyond the placed token
        player = self.board[row][col]
        count = 0

        r = row + dr
        c = col + dc

        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r][c] == player:
            count += 1
            r += dr
            c += dc

        return count

    def check_winner(self, row: int, col: int):
        # check the 4 line directions that matter in Connect 4
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        ]

        for dr, dc in directions:
            streak = 1
            streak += self.count_in_direction(row, col, dr, dc)
            streak += self.count_in_direction(row, col, -dr, -dc)

            if streak >= 4:
                return self.board[row][col]

        if not self.available_moves():
            return "Draw"

        return None

    def is_game_over(self) -> bool:
        return self.winner is not None

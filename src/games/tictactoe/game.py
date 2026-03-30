"""
Author: Priyansh Nayak
Description: Stores the Tic Tac Toe game state and rules
"""

class TicTacToe:

    WIN_LINES = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board = [" "] * 9
        self.current_player = "X"
        self.winner = None

    def available_moves(self) -> list[int]:
        return [index for index, value in enumerate(self.board) if value == " "]

    def make_move(self, position: int) -> bool:
        if self.winner is not None:
            return False

        if position < 0 or position >= 9 or self.board[position] != " ":
            return False

        self.board[position] = self.current_player
        self.winner = self.check_winner()

        if self.winner is None:
            self.current_player = "O" if self.current_player == "X" else "X"

        return True

    def check_winner(self) -> str | None:
        for a, b, c in self.WIN_LINES:
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]

        if " " not in self.board:
            return "Draw"

        return None

    def is_game_over(self) -> bool:
        return self.winner is not None

    def render(self) -> str:
        cells = [
            str(index + 1) if value == " " else value
            for index, value in enumerate(self.board)
        ]
        rows = [
            f" {cells[0]} | {cells[1]} | {cells[2]} ",
            f" {cells[3]} | {cells[4]} | {cells[5]} ",
            f" {cells[6]} | {cells[7]} | {cells[8]} ",
        ]
        return "\n---+---+---\n".join(rows)

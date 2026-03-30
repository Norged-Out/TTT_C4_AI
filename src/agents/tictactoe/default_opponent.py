"""
Author: Priyansh Nayak
Description: Simple default opponent for Tic Tac Toe
"""

from typing import Optional

from src.games.tictactoe.game import TicTacToe


def find_winning_move(game, player) -> Optional[int]:
    for move in game.available_moves():
        board_copy = game.board[:]
        board_copy[move] = player

        for a, b, c in TicTacToe.WIN_LINES:
            if board_copy[a] != " " and board_copy[a] == board_copy[b] == board_copy[c]:
                return move

    return None


def choose_default_move(game) -> int:
    # try to win
    move = find_winning_move(game, game.current_player)
    if move is not None:
        return move

    # try to block
    other = "O" if game.current_player == "X" else "X"
    move = find_winning_move(game, other)
    if move is not None:
        return move

    # simple fallback priority
    for move in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
        if move in game.available_moves():
            return move

    raise ValueError("No moves left for default opponent.")

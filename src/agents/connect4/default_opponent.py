"""
Author: Priyansh Nayak
Description: Simple default opponent for Connect 4
"""

import random

from src.games.connect4.game import Connect4


def find_winning_move(game, player):
    for col in game.available_moves():
        row = game.get_drop_row(col)
        if row is None:
            continue

        board_copy = [r[:] for r in game.board]
        board_copy[row][col] = player

        temp_game = Connect4()
        temp_game.board = board_copy

        if temp_game.check_winner(row, col) == player:
            return col

    return None


def choose_default_move(game):
    # win if possible
    move = find_winning_move(game, game.current_player)
    if move is not None:
        return move

    # otherwise block
    other = "O" if game.current_player == "X" else "X"
    move = find_winning_move(game, other)
    if move is not None:
        return move

    # random fallback keeps this weaker
    return random.choice(game.available_moves())

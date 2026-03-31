"""
Author: Priyansh Nayak
Description: Simple default opponent for Connect 4
"""

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
    # try to win immediately
    move = find_winning_move(game, game.current_player)
    if move is not None:
        return move

    # otherwise block the opponent's immediate win
    other = "O" if game.current_player == "X" else "X"
    move = find_winning_move(game, other)
    if move is not None:
        return move

    # simple fallback: prefer the center columns first
    for col in [3, 2, 4, 1, 5, 0, 6]:
        if col in game.available_moves():
            return col

    raise ValueError("No moves left for default opponent.")

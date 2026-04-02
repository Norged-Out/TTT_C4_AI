"""
Author: Priyansh Nayak
Description: Minimax agent for Tic Tac Toe
"""

from src.games.tictactoe.game import TicTacToe


def check_winner(board):
    # shared terminal check on a plain board list
    for a, b, c in TicTacToe.WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]

    if " " not in board:
        return "Draw"

    return None


def utility(winner, ai_player):
    # final score from the AI point of view
    if winner == ai_player:
        return 1

    if winner == "Draw":
        return 0

    return -1


def max_value(board, ai_player):
    # AI turn
    winner = check_winner(board)
    if winner is not None:
        return utility(winner, ai_player)

    best_score = -999

    for move in [i for i, value in enumerate(board) if value == " "]:
        board[move] = ai_player
        score = min_value(board, ai_player)
        board[move] = " "

        if score > best_score:
            best_score = score

    return best_score


def min_value(board, ai_player):
    # opponent turn
    winner = check_winner(board)
    if winner is not None:
        return utility(winner, ai_player)

    best_score = 999
    other = "O" if ai_player == "X" else "X"

    for move in [i for i, value in enumerate(board) if value == " "]:
        board[move] = other
        score = max_value(board, ai_player)
        board[move] = " "

        if score < best_score:
            best_score = score

    return best_score


def choose_minimax_move(game):
    # try every move once from the current state
    ai_player = game.current_player
    best_move = None
    best_score = -999

    for move in game.available_moves():
        board_copy = game.board[:]
        board_copy[move] = ai_player

        score = min_value(board_copy, ai_player)

        if score > best_score:
            best_score = score
            best_move = move

    if best_move is None:
        raise ValueError("No moves left for minimax.")

    return best_move

"""
Author: Priyansh Nayak
Description: Minimax agent for Tic Tac Toe
"""

from src.games.tictactoe.game import TicTacToe


def check_winner(board):
    # helper for terminal-test(state)
    for a, b, c in TicTacToe.WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]

    if " " not in board:
        return "Draw"

    return None


def utility(winner, ai_player):
    # utility(state)
    # score the final result from the AI's point of view
    if winner == ai_player:
        return 1

    if winner == "Draw":
        return 0

    return -1


def max_value(board, ai_player):
    # MAX-VALUE(state)
    winner = check_winner(board)
    if winner is not None:
        return utility(winner, ai_player)

    best_score = -999

    # actions(state) = all empty squares
    # result(state, action) = place mark, recurse, then undo move
    # AI turn: try to maximize the score
    for move in [i for i, value in enumerate(board) if value == " "]:
        board[move] = ai_player
        score = min_value(board, ai_player)
        board[move] = " "

        if score > best_score:
            best_score = score

    return best_score


def min_value(board, ai_player):
    # MIN-VALUE(state)
    winner = check_winner(board)
    if winner is not None:
        return utility(winner, ai_player)

    best_score = 999
    other = "O" if ai_player == "X" else "X"

    # actions(state) = all empty squares
    # result(state, action) = place mark, recurse, then undo move
    # opponent turn: assume they minimize the AI's score
    for move in [i for i, value in enumerate(board) if value == " "]:
        board[move] = other
        score = max_value(board, ai_player)
        board[move] = " "

        if score < best_score:
            best_score = score

    return best_score


def choose_minimax_move(game):
    # MINIMAX-DECISION(state)
    ai_player = game.current_player
    best_move = None
    best_score = -999

    # actions(state) = all legal moves from current game state
    # try every legal move and keep the one with the best final score
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

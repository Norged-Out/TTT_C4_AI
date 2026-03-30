"""
Author: Priyansh Nayak
Description: Alpha Beta agent for Tic Tac Toe
"""

from src.agents.tictactoe.minimax import check_winner, utility


def max_value_ab(board, ai_player, alpha, beta):
    # MAX-VALUE(state) with alpha-beta pruning
    winner = check_winner(board)
    if winner is not None:
        return utility(winner, ai_player)

    best_score = -999

    # actions(state) = all empty squares
    # result(state, action) = place mark, recurse, then undo move
    for move in [i for i, value in enumerate(board) if value == " "]:
        board[move] = ai_player
        score = min_value_ab(board, ai_player, alpha, beta)
        board[move] = " "

        if score > best_score:
            best_score = score

        # update alpha = best MAX score found so far
        if best_score > alpha:
            alpha = best_score

        # prune: MIN will avoid this branch anyway
        if alpha >= beta:
            break

    return best_score


def min_value_ab(board, ai_player, alpha, beta):
    # MIN-VALUE(state) with alpha-beta pruning
    winner = check_winner(board)
    if winner is not None:
        return utility(winner, ai_player)

    best_score = 999
    other = "O" if ai_player == "X" else "X"

    # actions(state) = all empty squares
    # result(state, action) = place mark, recurse, then undo move
    for move in [i for i, value in enumerate(board) if value == " "]:
        board[move] = other
        score = max_value_ab(board, ai_player, alpha, beta)
        board[move] = " "

        if score < best_score:
            best_score = score

        # update beta = best MIN score found so far
        if best_score < beta:
            beta = best_score

        # prune: MAX already has a better option elsewhere
        if alpha >= beta:
            break

    return best_score


def choose_alphabeta_move(game):
    # ALPHA-BETA-SEARCH(state)
    ai_player = game.current_player
    best_move = None
    best_score = -999
    alpha = -999
    beta = 999

    for move in game.available_moves():
        board_copy = game.board[:]
        board_copy[move] = ai_player

        score = min_value_ab(board_copy, ai_player, alpha, beta)

        if score > best_score:
            best_score = score
            best_move = move

        if best_score > alpha:
            alpha = best_score

    if best_move is None:
        raise ValueError("No moves left for alpha beta.")

    return best_move

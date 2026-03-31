"""
Author: Priyansh Nayak
Description: Alpha Beta agent for Connect 4
"""

import time

from src.agents.connect4.minimax import SearchTimeout, check_timeout, clone_game, utility


def max_value_ab(game, ai_player, alpha, beta, stats, depth):
    # MAX-VALUE(state) with alpha-beta pruning
    check_timeout(stats)
    stats["nodes_visited"] += 1
    if depth > stats["max_depth_reached"]:
        stats["max_depth_reached"] = depth

    if game.winner is not None:
        stats["terminal_states"] += 1
        return utility(game.winner, ai_player)

    best_score = -999

    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = min_value_ab(next_game, ai_player, alpha, beta, stats, depth + 1)

        if score > best_score:
            best_score = score

        if best_score > alpha:
            alpha = best_score

        if alpha >= beta:
            stats["prunes"] += 1
            break

    return best_score


def min_value_ab(game, ai_player, alpha, beta, stats, depth):
    # MIN-VALUE(state) with alpha-beta pruning
    check_timeout(stats)
    stats["nodes_visited"] += 1
    if depth > stats["max_depth_reached"]:
        stats["max_depth_reached"] = depth

    if game.winner is not None:
        stats["terminal_states"] += 1
        return utility(game.winner, ai_player)

    best_score = 999

    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = max_value_ab(next_game, ai_player, alpha, beta, stats, depth + 1)

        if score < best_score:
            best_score = score

        if best_score < beta:
            beta = best_score

        if alpha >= beta:
            stats["prunes"] += 1
            break

    return best_score


def choose_alphabeta_move(game, time_limit=None):
    # ALPHA-BETA-SEARCH(state)
    if game.winner is not None:
        raise ValueError("Cannot run alpha beta on a finished game.")

    ai_player = game.current_player
    best_move = None
    best_score = -999
    alpha = -999
    beta = 999
    stats = {
        "nodes_visited": 0,
        "terminal_states": 0,
        "max_depth_reached": 0,
        "prunes": 0,
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "deadline": None if time_limit is None else time.perf_counter() + time_limit,
    }

    start = time.perf_counter()

    try:
        for move in game.available_moves():
            next_game = clone_game(game)
            next_game.make_move(move)
            score = min_value_ab(next_game, ai_player, alpha, beta, stats, depth=1)

            if score > best_score:
                best_score = score
                best_move = move

            if best_score > alpha:
                alpha = best_score
    except SearchTimeout:
        stats["timed_out"] = True

    if best_move is None:
        stats["elapsed_seconds"] = time.perf_counter() - start
        stats.pop("deadline", None)
        raise SearchTimeout(stats)

    stats["chosen_move"] = best_move
    stats["chosen_score"] = best_score
    stats["elapsed_seconds"] = time.perf_counter() - start
    stats.pop("deadline", None)
    return best_move, stats

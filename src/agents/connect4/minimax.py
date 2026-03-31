"""
Author: Priyansh Nayak
Description: Minimax agent for Connect 4
"""

import time

from src.games.connect4.game import Connect4


class SearchTimeout(Exception):
    pass


def clone_game(game):
    copy = Connect4()
    copy.board = [row[:] for row in game.board]
    copy.current_player = game.current_player
    copy.winner = game.winner
    copy.last_move = game.last_move
    return copy


def utility(winner, ai_player):
    # utility(state)
    if winner == ai_player:
        return 1

    if winner == "Draw":
        return 0

    return -1


def check_timeout(stats):
    if stats["deadline"] is None:
        return

    if time.perf_counter() >= stats["deadline"]:
        raise SearchTimeout


def max_value(game, ai_player, stats, depth):
    # MAX-VALUE(state)
    check_timeout(stats)
    stats["nodes_visited"] += 1
    if depth > stats["max_depth_reached"]:
        stats["max_depth_reached"] = depth

    if game.winner is not None:
        stats["terminal_states"] += 1
        return utility(game.winner, ai_player)

    best_score = -999

    # actions(state) = all playable columns
    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = min_value(next_game, ai_player, stats, depth + 1)

        if score > best_score:
            best_score = score

    return best_score


def min_value(game, ai_player, stats, depth):
    # MIN-VALUE(state)
    check_timeout(stats)
    stats["nodes_visited"] += 1
    if depth > stats["max_depth_reached"]:
        stats["max_depth_reached"] = depth

    if game.winner is not None:
        stats["terminal_states"] += 1
        return utility(game.winner, ai_player)

    best_score = 999

    # actions(state) = all playable columns
    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = max_value(next_game, ai_player, stats, depth + 1)

        if score < best_score:
            best_score = score

    return best_score


def choose_minimax_move(game, time_limit=None):
    # MINIMAX-DECISION(state)
    if game.winner is not None:
        raise ValueError("Cannot run minimax on a finished game.")

    ai_player = game.current_player
    best_move = None
    best_score = -999
    stats = {
        "nodes_visited": 0,
        "terminal_states": 0,
        "max_depth_reached": 0,
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "deadline": None if time_limit is None else time.perf_counter() + time_limit,
    }

    start = time.perf_counter()

    try:
        for move in game.available_moves():
            next_game = clone_game(game)
            next_game.make_move(move)
            score = min_value(next_game, ai_player, stats, depth=1)

            if score > best_score:
                best_score = score
                best_move = move
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

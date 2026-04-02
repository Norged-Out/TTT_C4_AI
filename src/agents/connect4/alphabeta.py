"""
Author: Priyansh Nayak
Description: Alpha Beta agent for Connect 4
"""

import time

from src.agents.connect4.minimax import (
    INF,
    SearchTimeout,
    build_stats,
    check_timeout,
    clone_game,
    finish_stats,
    get_state_score,
    visit_state,
)


def max_value_ab(game, ai_player, alpha, beta, stats, depth):
    # AI turn with pruning
    check_timeout(stats)
    visit_state(stats, depth)
    score = get_state_score(game, ai_player, stats, depth)
    if score is not None:
        return score

    best_score = -999

    # try every legal move
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
    # opponent turn with pruning
    check_timeout(stats)
    visit_state(stats, depth)
    score = get_state_score(game, ai_player, stats, depth)
    if score is not None:
        return score

    best_score = 999

    # try every legal move
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
    # full alpha-beta search
    if game.winner is not None:
        raise ValueError("Cannot run alpha beta on a finished game.")

    ai_player = game.current_player
    best_move = None
    best_score = -999
    alpha = -999
    beta = 999
    stats = build_stats(time_limit=time_limit)
    stats["prunes"] = 0

    start = time.perf_counter()

    # root loop over all legal moves
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

    finish_stats(stats, start, best_move, best_score)
    return best_move, stats


def max_value_ab_limited(game, ai_player, alpha, beta, stats, depth, depth_limit):
    # AI turn for the limited version
    check_timeout(stats)
    visit_state(stats, depth)
    score = get_state_score(game, ai_player, stats, depth, depth_limit)
    if score is not None:
        return score

    best_score = -INF

    # try every legal move
    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = min_value_ab_limited(next_game, ai_player, alpha, beta, stats, depth + 1, depth_limit)

        if score > best_score:
            best_score = score

        if best_score > alpha:
            alpha = best_score

        if alpha >= beta:
            stats["prunes"] += 1
            break

    return best_score


def min_value_ab_limited(game, ai_player, alpha, beta, stats, depth, depth_limit):
    # opponent turn for the limited version
    check_timeout(stats)
    visit_state(stats, depth)
    score = get_state_score(game, ai_player, stats, depth, depth_limit)
    if score is not None:
        return score

    best_score = INF

    # try every legal move
    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = max_value_ab_limited(next_game, ai_player, alpha, beta, stats, depth + 1, depth_limit)

        if score < best_score:
            best_score = score

        if best_score < beta:
            beta = best_score

        if alpha >= beta:
            stats["prunes"] += 1
            break

    return best_score


def choose_alphabeta_move_limited(game, depth_limit=5, time_limit=None):
    # practical alpha-beta version
    if game.winner is not None:
        raise ValueError("Cannot run alpha beta on a finished game.")

    ai_player = game.current_player
    best_move = None
    best_score = -INF
    alpha = -INF
    beta = INF
    stats = build_stats(time_limit=time_limit, depth_limit=depth_limit)
    stats["prunes"] = 0

    start = time.perf_counter()

    # root loop over all legal moves
    try:
        for move in game.available_moves():
            next_game = clone_game(game)
            next_game.make_move(move)
            score = min_value_ab_limited(
                next_game,
                ai_player,
                alpha,
                beta,
                stats,
                depth=1,
                depth_limit=depth_limit,
            )

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

    finish_stats(stats, start, best_move, best_score)
    return best_move, stats

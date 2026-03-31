"""
Author: Priyansh Nayak
Description: Minimax agent for Connect 4
"""

import time

from src.games.connect4.game import Connect4


class SearchTimeout(Exception):
    pass


INF = 10**9


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


def score_window(window, ai_player):
    other = "O" if ai_player == "X" else "X"

    ai_count = window.count(ai_player)
    other_count = window.count(other)
    empty_count = window.count(" ")

    if ai_count == 4:
        return 100000

    if ai_count == 3 and empty_count == 1:
        return 100

    if ai_count == 2 and empty_count == 2:
        return 10

    if other_count == 3 and empty_count == 1:
        return -80

    if other_count == 4:
        return -100000

    return 0


def evaluate_board(game, ai_player):
    # heuristic evaluation for depth-limited search
    board = game.board
    score = 0

    # center column is usually stronger in Connect 4
    center_col = Connect4.COLS // 2
    center_values = [board[row][center_col] for row in range(Connect4.ROWS)]
    score += center_values.count(ai_player) * 6

    # horizontal windows
    for row in range(Connect4.ROWS):
        for col in range(Connect4.COLS - 3):
            window = [board[row][col + i] for i in range(4)]
            score += score_window(window, ai_player)

    # vertical windows
    for row in range(Connect4.ROWS - 3):
        for col in range(Connect4.COLS):
            window = [board[row + i][col] for i in range(4)]
            score += score_window(window, ai_player)

    # diagonal down-right windows
    for row in range(Connect4.ROWS - 3):
        for col in range(Connect4.COLS - 3):
            window = [board[row + i][col + i] for i in range(4)]
            score += score_window(window, ai_player)

    # diagonal down-left windows
    for row in range(Connect4.ROWS - 3):
        for col in range(3, Connect4.COLS):
            window = [board[row + i][col - i] for i in range(4)]
            score += score_window(window, ai_player)

    return score


def check_timeout(stats):
    if stats["deadline"] is None:
        return

    if time.perf_counter() >= stats["deadline"]:
        raise SearchTimeout


def visit_state(stats, depth):
    stats["nodes_visited"] += 1
    if depth > stats["max_depth_reached"]:
        stats["max_depth_reached"] = depth


def build_stats(time_limit=None, depth_limit=None):
    return {
        "nodes_visited": 0,
        "terminal_states": 0,
        "max_depth_reached": 0,
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "depth_limit": depth_limit,
        "cutoff_states": 0,
        "deadline": None if time_limit is None else time.perf_counter() + time_limit,
    }


def finish_stats(stats, start_time, best_move, best_score):
    stats["chosen_move"] = best_move
    stats["chosen_score"] = best_score
    stats["elapsed_seconds"] = time.perf_counter() - start_time
    stats.pop("deadline", None)
    if stats["depth_limit"] is None:
        stats.pop("depth_limit", None)
        stats.pop("cutoff_states", None)


def get_state_score(game, ai_player, stats, depth, depth_limit=None):
    if game.winner is not None:
        stats["terminal_states"] += 1
        return utility(game.winner, ai_player)

    if depth_limit is None or depth < depth_limit:
        return None

    stats["cutoff_states"] += 1
    return evaluate_board(game, ai_player)


def max_value_limited(game, ai_player, stats, depth, depth_limit):
    check_timeout(stats)
    visit_state(stats, depth)
    score = get_state_score(game, ai_player, stats, depth, depth_limit)
    if score is not None:
        return score

    best_score = -INF

    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = min_value_limited(next_game, ai_player, stats, depth + 1, depth_limit)

        if score > best_score:
            best_score = score

    return best_score


def min_value_limited(game, ai_player, stats, depth, depth_limit):
    check_timeout(stats)
    visit_state(stats, depth)
    score = get_state_score(game, ai_player, stats, depth, depth_limit)
    if score is not None:
        return score

    best_score = INF

    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = max_value_limited(next_game, ai_player, stats, depth + 1, depth_limit)

        if score < best_score:
            best_score = score

    return best_score


def max_value(game, ai_player, stats, depth):
    check_timeout(stats)
    visit_state(stats, depth)
    score = get_state_score(game, ai_player, stats, depth)
    if score is not None:
        return score

    best_score = -999
    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = min_value(next_game, ai_player, stats, depth + 1)

        if score > best_score:
            best_score = score

    return best_score


def min_value(game, ai_player, stats, depth):
    check_timeout(stats)
    visit_state(stats, depth)
    score = get_state_score(game, ai_player, stats, depth)
    if score is not None:
        return score

    best_score = 999
    for move in game.available_moves():
        next_game = clone_game(game)
        next_game.make_move(move)
        score = max_value(next_game, ai_player, stats, depth + 1)

        if score < best_score:
            best_score = score

    return best_score


def choose_minimax_move(game, time_limit=None):
    if game.winner is not None:
        raise ValueError("Cannot run minimax on a finished game.")

    ai_player = game.current_player
    best_move = None
    best_score = -999
    stats = build_stats(time_limit=time_limit)

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

    finish_stats(stats, start, best_move, best_score)
    return best_move, stats


def choose_minimax_move_limited(game, depth_limit=5, time_limit=None):
    if game.winner is not None:
        raise ValueError("Cannot run minimax on a finished game.")

    ai_player = game.current_player
    best_move = None
    best_score = -INF
    stats = build_stats(time_limit=time_limit, depth_limit=depth_limit)

    start = time.perf_counter()

    try:
        for move in game.available_moves():
            next_game = clone_game(game)
            next_game.make_move(move)
            score = min_value_limited(next_game, ai_player, stats, depth=1, depth_limit=depth_limit)

            if score > best_score:
                best_score = score
                best_move = move
    except SearchTimeout:
        stats["timed_out"] = True

    if best_move is None:
        stats["elapsed_seconds"] = time.perf_counter() - start
        stats.pop("deadline", None)
        raise SearchTimeout(stats)

    finish_stats(stats, start, best_move, best_score)
    return best_move, stats

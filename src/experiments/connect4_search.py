"""
Author: Priyansh Nayak
Description: Timed search benchmark for Connect 4 minimax and alpha beta
"""

from src.agents.connect4.alphabeta import choose_alphabeta_move, choose_alphabeta_move_limited
from src.agents.connect4.minimax import SearchTimeout, choose_minimax_move, choose_minimax_move_limited
from src.games.connect4.game import Connect4


def run_search(search_name, time_limit=None, depth_limit=None):
    # run one search from the empty board
    game = Connect4()

    try:
        # choose the right search function
        if search_name == "Minimax":
            if depth_limit is None:
                move, stats = choose_minimax_move(game, time_limit=time_limit)
            else:
                move, stats = choose_minimax_move_limited(game, depth_limit=depth_limit, time_limit=time_limit)
        elif search_name == "AlphaBeta":
            if depth_limit is None:
                move, stats = choose_alphabeta_move(game, time_limit=time_limit)
            else:
                move, stats = choose_alphabeta_move_limited(game, depth_limit=depth_limit, time_limit=time_limit)
        else:
            raise ValueError(f"Unknown search: {search_name}")

        stats["search"] = search_name
        stats["finished"] = True
        stats["timed_out"] = False
        stats["chosen_move"] = move
        return stats

    except SearchTimeout as e:
        # timed runs return partial stats
        if e.args:
            stats = e.args[0]
        else:
            stats = {}

        stats["search"] = search_name
        stats["finished"] = False
        stats["timed_out"] = True
        return stats


def run_connect4_search_benchmark(time_limit=30):
    # full search benchmark
    results = []

    for search_name in ["Minimax", "AlphaBeta"]:
        results.append(run_search(search_name, time_limit))
    return results


def run_connect4_limited_benchmark(depth_limit=5, time_limit=None):
    # depth-limited benchmark
    results = []

    for search_name in ["Minimax", "AlphaBeta"]:
        results.append(run_search(search_name, time_limit=time_limit, depth_limit=depth_limit))
    return results

"""
Author: Priyansh Nayak
Description: Timed search benchmark for Connect 4 minimax and alpha beta
"""

from src.agents.connect4.alphabeta import choose_alphabeta_move
from src.agents.connect4.minimax import SearchTimeout, choose_minimax_move
from src.games.connect4.game import Connect4


def run_search(search_name, time_limit):
    game = Connect4()

    try:
        if search_name == "Minimax":
            move, stats = choose_minimax_move(game, time_limit=time_limit)
        elif search_name == "AlphaBeta":
            move, stats = choose_alphabeta_move(game, time_limit=time_limit)
        else:
            raise ValueError(f"Unknown search: {search_name}")

        stats["search"] = search_name
        stats["finished"] = True
        stats["timed_out"] = False
        stats["chosen_move"] = move
        return stats

    except SearchTimeout as e:
        if e.args:
            stats = e.args[0]
        else:
            stats = {}

        stats["search"] = search_name
        stats["finished"] = False
        stats["timed_out"] = True
        return stats


def run_connect4_search_benchmark(time_limit=30):
    results = []
    for search_name in ["Minimax", "AlphaBeta"]:
        results.append(run_search(search_name, time_limit))
    return results

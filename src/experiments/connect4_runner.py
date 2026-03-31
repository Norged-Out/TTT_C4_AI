"""
Author: Priyansh Nayak
Description: Runs Connect 4 experiments and collects match results
"""

import time

from src.agents.connect4.alphabeta import choose_alphabeta_move_limited
from src.agents.connect4.default_opponent import choose_default_move
from src.agents.connect4.minimax import choose_minimax_move_limited
from src.games.connect4.game import Connect4


def get_agent_move(agent_name, game):
    if agent_name == "Default":
        return choose_default_move(game), None

    if agent_name == "Minimax":
        return choose_minimax_move_limited(game, depth_limit=5)

    if agent_name == "AlphaBeta":
        return choose_alphabeta_move_limited(game, depth_limit=5)

    raise ValueError(f"Unknown agent: {agent_name}")


def play_one_game(x_agent, o_agent):
    game = Connect4()
    move_count = 0
    x_time = 0.0
    o_time = 0.0
    x_nodes = 0
    o_nodes = 0

    while not game.is_game_over():
        current_agent = x_agent if game.current_player == "X" else o_agent
        start_time = time.perf_counter()
        move, stats = get_agent_move(current_agent, game)
        elapsed = time.perf_counter() - start_time

        if game.current_player == "X":
            x_time += elapsed
            if stats is not None:
                x_nodes += stats["nodes_visited"]
        else:
            o_time += elapsed
            if stats is not None:
                o_nodes += stats["nodes_visited"]

        game.make_move(move)
        move_count += 1

    return {
        "winner": game.winner,
        "moves": move_count,
        "x_time": x_time,
        "o_time": o_time,
        "x_nodes": x_nodes,
        "o_nodes": o_nodes,
    }


def run_matchup(x_agent, o_agent, num_games):
    wins_x = 0
    wins_o = 0
    draws = 0
    total_moves = 0
    total_x_time = 0.0
    total_o_time = 0.0
    total_x_nodes = 0
    total_o_nodes = 0

    print(f"Running Connect 4 matchup: {x_agent} vs {o_agent} ({num_games} games)")

    for i in range(num_games):
        result = play_one_game(x_agent, o_agent)

        total_moves += result["moves"]
        total_x_time += result["x_time"]
        total_o_time += result["o_time"]
        total_x_nodes += result["x_nodes"]
        total_o_nodes += result["o_nodes"]

        if result["winner"] == "X":
            wins_x += 1
        elif result["winner"] == "O":
            wins_o += 1
        else:
            draws += 1

        if (i + 1) % 5 == 0 or (i + 1) == num_games:
            print(
                f"  completed {i + 1}/{num_games} games"
                f" | X wins: {wins_x}"
                f" | O wins: {wins_o}"
                f" | Draws: {draws}"
            )

    return {
        "game": "Connect4",
        "x_agent": x_agent,
        "o_agent": o_agent,
        "games": num_games,
        "x_wins": wins_x,
        "o_wins": wins_o,
        "draws": draws,
        "x_win_rate": wins_x / num_games,
        "o_win_rate": wins_o / num_games,
        "draw_rate": draws / num_games,
        "avg_moves": total_moves / num_games,
        "avg_x_time": total_x_time / num_games,
        "avg_o_time": total_o_time / num_games,
        "avg_x_nodes": total_x_nodes / num_games,
        "avg_o_nodes": total_o_nodes / num_games,
    }


def run_experiments(num_games=10):
    results = []
    agents = ["Default", "Minimax", "AlphaBeta"]

    print("Starting Connect 4 experiments")

    for agent in agents:
        results.append(run_matchup(agent, "Default", num_games))
        if agent != "Default":
            results.append(run_matchup("Default", agent, num_games))

    pairings = [
        ("Minimax", "AlphaBeta"),
        ("AlphaBeta", "Minimax"),
        ("Minimax", "Minimax"),
        ("AlphaBeta", "AlphaBeta"),
    ]

    for x_agent, o_agent in pairings:
        results.append(run_matchup(x_agent, o_agent, num_games))

    print("Finished Connect 4 experiments")
    return results

"""
Author: Priyansh Nayak
Description: Runs Tic Tac Toe experiments and collects match results
"""

import time

from src.agents.tictactoe.alphabeta import choose_alphabeta_move
from src.agents.tictactoe.default_opponent import choose_default_move
from src.agents.tictactoe.minimax import choose_minimax_move
from src.games.tictactoe.game import TicTacToe


def get_agent_move(agent_name, game):
    if agent_name == "Default":
        return choose_default_move(game)

    if agent_name == "Minimax":
        return choose_minimax_move(game)

    if agent_name == "AlphaBeta":
        return choose_alphabeta_move(game)

    raise ValueError(f"Unknown agent: {agent_name}")


def play_one_game(x_agent, o_agent):
    game = TicTacToe()
    move_count = 0
    x_time = 0.0
    o_time = 0.0

    while not game.is_game_over():
        current_agent = x_agent if game.current_player == "X" else o_agent
        start_time = time.perf_counter()
        move = get_agent_move(current_agent, game)
        elapsed = time.perf_counter() - start_time

        if game.current_player == "X":
            x_time += elapsed
        else:
            o_time += elapsed

        game.make_move(move)
        move_count += 1

    return {
        "winner": game.winner,
        "moves": move_count,
        "x_time": x_time,
        "o_time": o_time,
    }


def run_matchup(x_agent, o_agent, num_games):
    wins_x = 0
    wins_o = 0
    draws = 0
    total_moves = 0
    total_x_time = 0.0
    total_o_time = 0.0

    print(f"Running matchup: {x_agent} vs {o_agent} ({num_games} games)")

    for i in range(num_games):
        result = play_one_game(x_agent, o_agent)

        total_moves += result["moves"]
        total_x_time += result["x_time"]
        total_o_time += result["o_time"]

        if result["winner"] == "X":
            wins_x += 1
        elif result["winner"] == "O":
            wins_o += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0 or (i + 1) == num_games:
            print(
                f"  completed {i + 1}/{num_games} games"
                f" | X wins: {wins_x}"
                f" | O wins: {wins_o}"
                f" | Draws: {draws}"
            )

    return {
        "game": "TicTacToe",
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
    }


def run_experiments(num_games=10):
    results = []
    agents = ["Default", "Minimax", "AlphaBeta"]

    print("Starting Tic Tac Toe experiments")

    # each agent against default opponent
    for agent in agents:
        results.append(run_matchup(agent, "Default", num_games))
        if agent != "Default":
            results.append(run_matchup("Default", agent, num_games))

    # pairwise comparisons between stronger agents
    pairings = [
        ("Minimax", "AlphaBeta"),
        ("AlphaBeta", "Minimax"),
        ("Minimax", "Minimax"),
        ("AlphaBeta", "AlphaBeta"),
    ]

    for x_agent, o_agent in pairings:
        results.append(run_matchup(x_agent, o_agent, num_games))

    print("Finished Tic Tac Toe experiments")

    return results

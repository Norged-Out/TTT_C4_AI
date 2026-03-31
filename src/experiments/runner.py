"""
Author: Priyansh Nayak
Description: Runs Tic Tac Toe experiments and collects match results
"""

import time

from src.agents.tictactoe.alphabeta import choose_alphabeta_move
from src.agents.tictactoe.default_opponent import choose_default_move
from src.agents.tictactoe.dqn import choose_dqn_move, train_dqn
from src.agents.tictactoe.minimax import choose_minimax_move
from src.agents.tictactoe.q_learning import choose_q_move, train_q_learning
from src.games.tictactoe.game import TicTacToe


def get_agent_move(agent_name, game, q_table=None, dqn_model=None):
    # route each agent name to its move-selection function
    if agent_name == "Default":
        return choose_default_move(game)

    if agent_name == "Minimax":
        return choose_minimax_move(game)

    if agent_name == "AlphaBeta":
        return choose_alphabeta_move(game)

    if agent_name == "QLearning":
        if q_table is None:
            raise ValueError("Q-learning agent needs a trained q_table.")
        return choose_q_move(game, q_table)

    if agent_name == "DQN":
        if dqn_model is None:
            raise ValueError("DQN agent needs a trained model.")
        return choose_dqn_move(game, dqn_model)

    raise ValueError(f"Unknown agent: {agent_name}")


def play_one_game(x_agent, o_agent, q_table=None, dqn_model=None):
    game = TicTacToe()
    move_count = 0
    x_time = 0.0
    o_time = 0.0

    # play until one side wins or the board fills up
    while not game.is_game_over():
        current_agent = x_agent if game.current_player == "X" else o_agent
        start_time = time.perf_counter()
        move = get_agent_move(current_agent, game, q_table=q_table, dqn_model=dqn_model)
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


def run_matchup(x_agent, o_agent, num_games, q_table=None, dqn_model=None):
    wins_x = 0
    wins_o = 0
    draws = 0
    total_moves = 0
    total_x_time = 0.0
    total_o_time = 0.0

    print(f"Running matchup: {x_agent} vs {o_agent} ({num_games} games)")

    # repeat the same pairing many times and average the results
    for i in range(num_games):
        result = play_one_game(x_agent, o_agent, q_table=q_table, dqn_model=dqn_model)

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
    agents = ["Default", "Minimax", "AlphaBeta", "QLearning", "DQN"]

    print("Starting Tic Tac Toe experiments")
    print("Training Q-learning agent for experiment run")
    q_table = train_q_learning()
    print("Training DQN agent for experiment run")
    dqn_model = train_dqn()

    # assignment-style baseline comparisons against the default opponent
    for agent in agents:
        results.append(run_matchup(agent, "Default", num_games, q_table=q_table, dqn_model=dqn_model))
        if agent != "Default":
            results.append(run_matchup("Default", agent, num_games, q_table=q_table, dqn_model=dqn_model))

    # pairwise comparisons between the stronger agents
    pairings = [
        ("Minimax", "AlphaBeta"),
        ("AlphaBeta", "Minimax"),
        ("Minimax", "Minimax"),
        ("AlphaBeta", "AlphaBeta"),
        ("QLearning", "Minimax"),
        ("Minimax", "QLearning"),
        ("QLearning", "AlphaBeta"),
        ("AlphaBeta", "QLearning"),
        ("QLearning", "QLearning"),
        ("DQN", "Minimax"),
        ("Minimax", "DQN"),
        ("DQN", "AlphaBeta"),
        ("AlphaBeta", "DQN"),
        ("DQN", "QLearning"),
        ("QLearning", "DQN"),
        ("DQN", "DQN"),
    ]

    for x_agent, o_agent in pairings:
        results.append(run_matchup(x_agent, o_agent, num_games, q_table=q_table, dqn_model=dqn_model))

    print("Finished Tic Tac Toe experiments")

    return results

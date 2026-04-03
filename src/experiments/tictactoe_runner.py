"""
Author: Priyansh Nayak
Description: Runs Tic Tac Toe experiments and collects match results
"""

import time

from src.games.tictactoe.game import TicTacToe


def get_agent_move(agent_name, game, q_table=None, dqn_model=None):
    # pick the move function from the agent name
    if agent_name == "Default":
        from src.agents.tictactoe.default_opponent import choose_default_move
        return choose_default_move(game)

    if agent_name == "Minimax":
        from src.agents.tictactoe.minimax import choose_minimax_move
        return choose_minimax_move(game)

    if agent_name == "AlphaBeta":
        from src.agents.tictactoe.alphabeta import choose_alphabeta_move
        return choose_alphabeta_move(game)

    if agent_name == "QLearning":
        from src.agents.tictactoe.q_learning import choose_q_move
        if q_table is None:
            raise ValueError("Q-learning agent needs a trained q_table.")
        return choose_q_move(game, q_table)

    if agent_name == "DQN":
        from src.agents.tictactoe.dqn import choose_dqn_move
        if dqn_model is None:
            raise ValueError("DQN agent needs a trained model.")
        return choose_dqn_move(game, dqn_model)

    raise ValueError(f"Unknown agent: {agent_name}")


def play_one_game(first_agent, second_agent, q_table=None, dqn_model=None):
    game = TicTacToe()
    move_count = 0
    first_time = 0.0
    second_time = 0.0

    while not game.is_game_over():
        # X uses first_agent, O uses second_agent
        current_agent = first_agent if game.current_player == "X" else second_agent
        start_time = time.perf_counter()
        move = get_agent_move(current_agent, game, q_table=q_table, dqn_model=dqn_model)
        elapsed = time.perf_counter() - start_time

        if game.current_player == "X":
            first_time += elapsed
        else:
            second_time += elapsed

        game.make_move(move)
        move_count += 1

    return {
        "winner": game.winner,
        "moves": move_count,
        "first_time": first_time,
        "second_time": second_time,
    }


def run_matchup(player1_agent, player2_agent, num_games, q_table=None, dqn_model=None):
    player1_wins = 0
    player2_wins = 0
    draws = 0
    total_moves = 0
    total_player1_time = 0.0
    total_player2_time = 0.0

    print(f"Running matchup: {player1_agent} vs {player2_agent} ({num_games} games)")

    for i in range(num_games):
        # alternate who starts to make this fair
        if i % 2 == 0:
            first_agent = player1_agent
            second_agent = player2_agent
        else:
            first_agent = player2_agent
            second_agent = player1_agent

        result = play_one_game(first_agent, second_agent, q_table=q_table, dqn_model=dqn_model)

        total_moves += result["moves"]
        if first_agent == player1_agent:
            total_player1_time += result["first_time"]
            total_player2_time += result["second_time"]
        else:
            total_player1_time += result["second_time"]
            total_player2_time += result["first_time"]

        if result["winner"] == "Draw":
            draws += 1
        elif first_agent == player1_agent and result["winner"] == "X":
            player1_wins += 1
        elif first_agent == player1_agent and result["winner"] == "O":
            player2_wins += 1
        elif first_agent == player2_agent and result["winner"] == "X":
            player2_wins += 1
        else:
            player1_wins += 1

        if (i + 1) % 10 == 0 or (i + 1) == num_games:
            print(
                f"  completed {i + 1}/{num_games} games"
                f" | Player 1 wins: {player1_wins}"
                f" | Player 2 wins: {player2_wins}"
                f" | Draws: {draws}"
            )

    return {
        "game": "TicTacToe",
        "player1_agent": player1_agent,
        "player2_agent": player2_agent,
        "games": num_games,
        "player1_starts": (num_games + 1) // 2,
        "player2_starts": num_games // 2,
        "player1_wins": player1_wins,
        "player2_wins": player2_wins,
        "draws": draws,
        "player1_win_rate": player1_wins / num_games,
        "player2_win_rate": player2_wins / num_games,
        "draw_rate": draws / num_games,
        "avg_moves": total_moves / num_games,
        "avg_player1_time": total_player1_time / num_games,
        "avg_player2_time": total_player2_time / num_games,
    }


def run_experiments(num_games=10):
    results = []
    agents = ["Default", "Minimax", "AlphaBeta", "QLearning", "DQN"]

    # load RL agents once for the whole run
    print("Starting Tic Tac Toe experiments")
    print("Training Q-learning agent for experiment run")
    from src.agents.tictactoe.q_learning import train_q_learning
    q_table = train_q_learning()
    print("Training DQN agent for experiment run")
    from src.agents.tictactoe.dqn import train_dqn
    dqn_model = train_dqn()

    for agent in agents:
        results.append(run_matchup(agent, "Default", num_games, q_table=q_table, dqn_model=dqn_model))
        if agent != "Default":
            results.append(run_matchup("Default", agent, num_games, q_table=q_table, dqn_model=dqn_model))

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

    for player1_agent, player2_agent in pairings:
        results.append(run_matchup(player1_agent, player2_agent, num_games, q_table=q_table, dqn_model=dqn_model))

    print("Finished Tic Tac Toe experiments")

    return results

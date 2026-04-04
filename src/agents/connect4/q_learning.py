"""
Author: Priyansh Nayak
Description: Tabular Q-learning agent for Connect 4
"""

import os
import pickle
import random
from collections import deque

from tqdm import tqdm

from src.experiments.training_log import write_training_log
from src.games.connect4.game import Connect4


Q_TABLE_PATH = os.path.join("models", "connect4_q_table.pkl")
TRAINING_LOG_PATH = os.path.join("results", "training", "connect4_q_learning.csv")

def get_state_key(game):
    # board plus turn makes the state key
    board_text = "".join("".join(row) for row in game.board)
    return (board_text, game.current_player)


def get_q_value(q_table, state, action):
    # unseen pairs start at zero
    return q_table.get((state, action), 0.0)


def choose_q_move(game, q_table):
    # pick the move with the best learned value
    state = get_state_key(game)
    moves = game.available_moves()

    best_move = None
    best_value = -999999.0

    for move in moves:
        value = get_q_value(q_table, state, move)
        if value > best_value:
            best_value = value
            best_move = move

    if best_move is None:
        raise ValueError("No moves left for Q-learning.")

    return best_move


def choose_epsilon_greedy_move(game, q_table, epsilon):
    # random sometimes, greedy otherwise
    moves = game.available_moves()

    if random.random() < epsilon:
        return random.choice(moves)

    return choose_q_move(game, q_table)


def reward_from_winner(winner, player):
    # simple terminal reward setup
    if winner == player:
        return 1.0

    if winner == "Draw":
        return 0.0

    return -1.0


def update_q_value(q_table, state, action, reward, next_state, next_moves, alpha, gamma):
    # standard tabular Q update
    old_value = get_q_value(q_table, state, action)

    if not next_moves or next_state is None:
        future_value = 0.0
    else:
        future_value = max(get_q_value(q_table, next_state, move) for move in next_moves)

    new_value = old_value + alpha * (reward + gamma * future_value - old_value)
    q_table[(state, action)] = new_value


def train_q_learning(
    episodes=50000,
    progress_callback=None,
    model_path=Q_TABLE_PATH,
    force_retrain=False,
    log_path=TRAINING_LOG_PATH,
):
    # load the saved table unless we want a fresh run
    if os.path.exists(model_path) and not force_retrain:
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # learning settings
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.3
    min_epsilon = 0.05

    q_table = {}
    recent_results = deque(maxlen=500)
    training_rows = []

    for episode in tqdm(range(episodes), desc="Connect4 Q-learning", unit="episode"):
        # alternate which side the learner plays
        game = Connect4()
        q_player = "X" if episode % 2 == 0 else "O"
        history = []

        while not game.is_game_over():
            if game.current_player != q_player:
                # let the random training opponent play
                game.make_move(random.choice(game.available_moves()))
                continue

            # state before the move
            state = get_state_key(game)
            action = choose_epsilon_greedy_move(game, q_table, epsilon)

            game.make_move(action)

            # state after the move
            next_state = get_state_key(game)
            next_moves = game.available_moves()
            reward = 0.0

            if game.is_game_over():
                reward = reward_from_winner(game.winner, q_player)

            # immediate update for this move
            update_q_value(q_table, state, action, reward, next_state, next_moves, alpha, gamma)
            history.append((state, action))

        final_reward = reward_from_winner(game.winner, q_player)

        if game.winner == q_player:
            replay_history = history[:-1]
        else:
            replay_history = history

        # push the final result back through earlier moves
        for state, action in reversed(replay_history):
            update_q_value(q_table, state, action, final_reward, None, [], alpha, gamma)

        # slowly reduce random exploration
        epsilon = max(min_epsilon, epsilon * 0.99995)
        recent_results.append((game.winner, q_player, final_reward))

        if (episode + 1) % 250 == 0 or (episode + 1) == episodes:
            # save a small training log
            total_recent = len(recent_results)
            training_rows.append({
                "episode": episode + 1,
                "epsilon": epsilon,
                "agent_win_rate": sum(1 for winner, player, _ in recent_results if winner == player) / total_recent,
                "opponent_win_rate": sum(1 for winner, player, _ in recent_results if winner not in {player, "Draw"}) / total_recent,
                "draw_rate": sum(1 for winner, _, _ in recent_results if winner == "Draw") / total_recent,
                "avg_reward": sum(reward for _, _, reward in recent_results) / total_recent,
            })

        if progress_callback is not None:
            progress_callback(episode + 1, episodes)

    # save the trained table
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(q_table, f)

    # save the training curve too
    write_training_log(log_path, training_rows)

    return q_table

"""
Author: Priyansh Nayak
Description: Tabular Q-learning agent for Connect 4
"""

import os
import pickle
import random

from tqdm import tqdm

from src.games.connect4.game import Connect4


Q_TABLE_PATH = os.path.join("models", "connect4_q_table.pkl")


def choose_random_move(game):
    return random.choice(game.available_moves())


def get_state_key(game):
    board_text = "".join("".join(row) for row in game.board)
    return (board_text, game.current_player)


def get_q_value(q_table, state, action):
    return q_table.get((state, action), 0.0)


def choose_q_move(game, q_table):
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
    moves = game.available_moves()

    if random.random() < epsilon:
        return random.choice(moves)

    return choose_q_move(game, q_table)


def reward_from_winner(winner, player):
    if winner == player:
        return 1.0

    if winner == "Draw":
        return 0.0

    return -1.0


def update_q_value(q_table, state, action, reward, next_state, next_moves, alpha, gamma):
    old_value = get_q_value(q_table, state, action)

    if not next_moves or next_state is None:
        future_value = 0.0
    else:
        future_value = max(get_q_value(q_table, next_state, move) for move in next_moves)

    new_value = old_value + alpha * (reward + gamma * future_value - old_value)
    q_table[(state, action)] = new_value


def train_q_learning(episodes=20000, progress_callback=None, model_path=Q_TABLE_PATH, force_retrain=False):
    if os.path.exists(model_path) and not force_retrain:
        with open(model_path, "rb") as f:
            return pickle.load(f)

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.3
    epsilon_decay = 0.99995
    min_epsilon = 0.05

    q_table = {}

    for episode in tqdm(range(episodes), desc="Connect4 Q-learning", unit="episode"):
        game = Connect4()
        q_player = "X" if episode % 2 == 0 else "O"
        history = []

        while not game.is_game_over():
            if game.current_player != q_player:
                game.make_move(choose_random_move(game))
                continue

            state = get_state_key(game)
            action = choose_epsilon_greedy_move(game, q_table, epsilon)

            game.make_move(action)

            next_state = get_state_key(game)
            next_moves = game.available_moves()
            reward = 0.0

            if game.is_game_over():
                reward = reward_from_winner(game.winner, q_player)

            update_q_value(q_table, state, action, reward, next_state, next_moves, alpha, gamma)
            history.append((state, action))

        final_reward = reward_from_winner(game.winner, q_player)

        if game.winner == q_player:
            replay_history = history[:-1]
        else:
            replay_history = history

        for state, action in reversed(replay_history):
            update_q_value(q_table, state, action, final_reward, None, [], alpha, gamma)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if progress_callback is not None:
            progress_callback(episode + 1, episodes)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(q_table, f)

    return q_table

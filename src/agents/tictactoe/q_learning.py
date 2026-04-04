"""
Author: Priyansh Nayak
Description: Tabular Q-learning agent for Tic Tac Toe
"""

import os
import pickle
import random
from collections import deque

from tqdm import tqdm

from src.experiments.training_log import write_training_log
from src.games.tictactoe.game import TicTacToe


Q_TABLE_PATH = os.path.join("models", "tictactoe_q_table.pkl")
TRAINING_LOG_PATH = os.path.join("results", "training", "tictactoe_q_learning.csv")


def get_state_key(game):
    # board plus turn makes the state key
    return ("".join(game.board), game.current_player)


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

    if not next_moves:
        future_value = 0.0
    else:
        future_value = max(get_q_value(q_table, next_state, move) for move in next_moves)

    new_value = old_value + alpha * (reward + gamma * future_value - old_value)
    q_table[(state, action)] = new_value


def train_q_learning(
    episodes=20000,
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
    recent_winners = deque(maxlen=200)
    training_rows = []

    for episode in tqdm(range(episodes), desc="Q-learning", unit="episode"):
        # one episode is one full game
        game = TicTacToe()
        history = []

        while not game.is_game_over():
            # state before the move
            state = get_state_key(game)
            action = choose_epsilon_greedy_move(game, q_table, epsilon)
            player = game.current_player

            game.make_move(action)

            # state after the move
            next_state = get_state_key(game)
            next_moves = game.available_moves()
            reward = 0.0

            if game.is_game_over():
                reward = reward_from_winner(game.winner, player)

            # immediate update for this move
            update_q_value(q_table, state, action, reward, next_state, next_moves, alpha, gamma)
            history.append((state, action, player))

        # push the final result back through earlier moves
        for state, action, player in reversed(history[:-1]):
            reward = reward_from_winner(game.winner, player)
            update_q_value(q_table, state, action, reward, None, [], alpha, gamma)

        # slowly reduce random exploration
        epsilon = max(min_epsilon, epsilon * 0.9998)
        recent_winners.append(game.winner)

        if (episode + 1) % 100 == 0 or (episode + 1) == episodes:
            # save a small training log
            total_recent = len(recent_winners)
            training_rows.append({
                "episode": episode + 1,
                "epsilon": epsilon,
                "x_win_rate": sum(1 for winner in recent_winners if winner == "X") / total_recent,
                "o_win_rate": sum(1 for winner in recent_winners if winner == "O") / total_recent,
                "draw_rate": sum(1 for winner in recent_winners if winner == "Draw") / total_recent,
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

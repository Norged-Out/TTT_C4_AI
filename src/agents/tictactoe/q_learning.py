"""
Author: Priyansh Nayak
Description: Tabular Q-learning agent for Tic Tac Toe
"""

import random

from src.games.tictactoe.game import TicTacToe


def get_state_key(game):
    # include current player so X-turn and O-turn are different states
    return ("".join(game.board), game.current_player)


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

    if not next_moves:
        future_value = 0.0
    else:
        future_value = max(get_q_value(q_table, next_state, move) for move in next_moves)

    # Q(s,a) = old + alpha * (reward + gamma * max Q(s',a') - old)
    new_value = old_value + alpha * (reward + gamma * future_value - old_value)
    q_table[(state, action)] = new_value


def train_q_learning(episodes=20000):
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.3
    epsilon_decay = 0.9998
    min_epsilon = 0.05

    q_table = {}

    print(f"Training Q-learning for {episodes} episodes")
    print(
        f"alpha={alpha} | gamma={gamma} | epsilon={epsilon}"
        f" | epsilon_decay={epsilon_decay} | min_epsilon={min_epsilon}"
    )

    for episode in range(episodes):
        game = TicTacToe()
        history = []

        while not game.is_game_over():
            state = get_state_key(game)
            action = choose_epsilon_greedy_move(game, q_table, epsilon)
            player = game.current_player

            game.make_move(action)

            next_state = get_state_key(game)
            next_moves = game.available_moves()
            reward = 0.0

            if game.is_game_over():
                reward = reward_from_winner(game.winner, player)

            update_q_value(q_table, state, action, reward, next_state, next_moves, alpha, gamma)
            history.append((state, action, player))

        # after game ends, push final reward backward to earlier moves by same player
        for state, action, player in reversed(history[:-1]):
            reward = reward_from_winner(game.winner, player)
            update_q_value(q_table, state, action, reward, None, [], alpha, gamma)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 500 == 0 or (episode + 1) == episodes:
            print(
                f"  completed {episode + 1}/{episodes} episodes"
                f" | epsilon={epsilon:.3f}"
                f" | q_entries={len(q_table)}"
            )

    print("Finished Q-learning training")
    return q_table

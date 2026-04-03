"""
Author: Priyansh Nayak
Description: Simple DQN agent for Tic Tac Toe
"""

import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.experiments.training_log import write_training_log
from src.games.tictactoe.game import TicTacToe


DQN_MODEL_PATH = os.path.join("models", "tictactoe_dqn.pt")
TRAINING_LOG_PATH = os.path.join("results", "training", "tictactoe_dqn.csv")


class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # small network is enough here
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )

    def forward(self, x):
        return self.model(x)


def state_to_tensor(game):
    # current player is 1, other player is -1
    current = game.current_player
    other = "O" if current == "X" else "X"

    values = []
    for cell in game.board:
        if cell == current:
            values.append(1.0)
        elif cell == other:
            values.append(-1.0)
        else:
            values.append(0.0)

    return torch.tensor(values, dtype=torch.float32)


def choose_dqn_move(game, model):
    # only pick from legal moves
    state = state_to_tensor(game).unsqueeze(0)
    legal_moves = game.available_moves()

    with torch.no_grad():
        q_values = model(state)[0]

    best_move = None
    best_value = -999999.0

    for move in legal_moves:
        value = float(q_values[move].item())
        if value > best_value:
            best_value = value
            best_move = move

    if best_move is None:
        raise ValueError("No moves left for DQN.")

    return best_move


def choose_epsilon_greedy_move(game, model, epsilon):
    # random sometimes, greedy otherwise
    legal_moves = game.available_moves()

    if random.random() < epsilon:
        return random.choice(legal_moves)

    return choose_dqn_move(game, model)


def reward_from_winner(winner, player):
    # simple terminal reward setup
    if winner == player:
        return 1.0

    if winner == "Draw":
        return 0.0

    return -1.0


def train_dqn(
    episodes=5000,
    progress_callback=None,
    model_path=DQN_MODEL_PATH,
    force_retrain=False,
    log_path=TRAINING_LOG_PATH,
):
    # load the saved model unless we want a fresh run
    gamma = 0.9
    epsilon = 0.3
    epsilon_decay = 0.9995
    min_epsilon = 0.05
    batch_size = 64
    replay_size = 10000

    if os.path.exists(model_path) and not force_retrain:
        model = DQNNet()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    model = DQNNet()

    # training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    replay = deque(maxlen=replay_size)
    recent_winners = deque(maxlen=200)
    training_rows = []

    for episode in tqdm(range(episodes), desc="DQN", unit="episode"):
        # one episode is one full game
        game = TicTacToe()

        while not game.is_game_over():
            # state before the move
            player = game.current_player
            state = state_to_tensor(game)
            move = choose_epsilon_greedy_move(game, model, epsilon)

            game.make_move(move)

            # state after the move
            reward = 0.0
            done = game.is_game_over()

            if done:
                reward = reward_from_winner(game.winner, player)
                next_state = torch.zeros(9, dtype=torch.float32)
                next_moves = []
            else:
                next_state = state_to_tensor(game)
                next_moves = game.available_moves()

            replay.append((state, move, reward, next_state, next_moves, done))

            if len(replay) >= batch_size:
                # train from random replay samples
                batch = random.sample(replay, batch_size)

                states = torch.stack([item[0] for item in batch])
                actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
                rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
                next_states = torch.stack([item[3] for item in batch])
                dones = torch.tensor([item[5] for item in batch], dtype=torch.float32)

                # current Q values for the chosen moves
                q_values = model(states)
                chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # next-state values for the target
                    next_q_values = model(next_states)
                    next_best = []

                    for i, (_, _, _, _, legal_moves, done_flag) in enumerate(batch):
                        if done_flag or not legal_moves:
                            next_best.append(0.0)
                        else:
                            best_value = max(float(next_q_values[i][m].item()) for m in legal_moves)
                            next_best.append(best_value)

                    next_best_tensor = torch.tensor(next_best, dtype=torch.float32)
                    targets = rewards + gamma * next_best_tensor * (1.0 - dones)

                # normal gradient step
                loss = loss_fn(chosen_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # slowly reduce random exploration
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        recent_winners.append(game.winner)

        if (episode + 1) % 100 == 0 or (episode + 1) == episodes:
            # keep a small convergence log for the report
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

    # save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # save the training curve too
    write_training_log(log_path, training_rows)

    model.eval()
    return model

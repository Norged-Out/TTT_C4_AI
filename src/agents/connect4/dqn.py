"""
Author: Priyansh Nayak
Description: Simple DQN agent for Connect 4
"""

import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.games.connect4.game import Connect4
from src.agents.connect4.q_learning import choose_random_move, reward_from_winner


DQN_MODEL_PATH = os.path.join("models", "connect4_dqn.pt")


class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        return self.model(x)


def state_to_tensor(game):
    current = game.current_player
    other = "O" if current == "X" else "X"
    values = []

    for row in game.board:
        for cell in row:
            if cell == current:
                values.append(1.0)
            elif cell == other:
                values.append(-1.0)
            else:
                values.append(0.0)

    return torch.tensor(values, dtype=torch.float32)


def choose_dqn_move(game, model):
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
    legal_moves = game.available_moves()

    if random.random() < epsilon:
        return random.choice(legal_moves)

    return choose_dqn_move(game, model)


def train_dqn(episodes=20000, progress_callback=None, model_path=DQN_MODEL_PATH, force_retrain=False):
    gamma = 0.9
    epsilon = 0.3
    epsilon_decay = 0.99995
    min_epsilon = 0.05
    batch_size = 64
    replay_size = 20000

    model = DQNNet()

    if os.path.exists(model_path) and not force_retrain:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    replay = deque(maxlen=replay_size)

    for episode in tqdm(range(episodes), desc="Connect4 DQN", unit="episode"):
        game = Connect4()
        dqn_player = "X" if episode % 2 == 0 else "O"

        while not game.is_game_over():
            if game.current_player != dqn_player:
                game.make_move(choose_random_move(game))
                continue

            state = state_to_tensor(game)
            move = choose_epsilon_greedy_move(game, model, epsilon)
            game.make_move(move)

            reward = 0.0
            done = game.is_game_over()

            if done:
                reward = reward_from_winner(game.winner, dqn_player)
                next_state = torch.zeros(42, dtype=torch.float32)
                next_moves = []
            else:
                next_state = state_to_tensor(game)
                next_moves = game.available_moves()

            replay.append((state, move, reward, next_state, next_moves, done))

            if len(replay) < batch_size:
                continue

            batch = random.sample(replay, batch_size)
            states = torch.stack([item[0] for item in batch])
            actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
            rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            next_states = torch.stack([item[3] for item in batch])
            dones = torch.tensor([item[5] for item in batch], dtype=torch.float32)

            q_values = model(states)
            chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = model(next_states)
                next_best = []

                for i, (_, _, _, _, legal_moves, done_flag) in enumerate(batch):
                    if done_flag or not legal_moves:
                        next_best.append(0.0)
                        continue

                    best_value = max(float(next_q_values[i][m].item()) for m in legal_moves)
                    next_best.append(best_value)

                next_best_tensor = torch.tensor(next_best, dtype=torch.float32)
                targets = rewards - gamma * next_best_tensor * (1.0 - dones)

            loss = loss_fn(chosen_q, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if progress_callback is not None:
            progress_callback(episode + 1, episodes)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    model.eval()
    return model

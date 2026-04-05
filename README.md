# Tic Tac Toe and Connect 4 AI

Implementation and comparative analysis of adversarial search and reinforcement learning methods for Tic-Tac-Toe and Connect 4.

This project was developed for **CS7IS2 – Artificial Intelligence** and includes:

* Tic-Tac-Toe and Connect 4 game environments
* Default baseline opponents
* Minimax
* Alpha-Beta Pruning
* Tabular Q-Learning
* Deep Q-Network (DQN)
* Automated experiments
* Performance analysis and visualisation
* Interactive Pygame demos

---

## Overview

The objective of this project is to compare classical adversarial search methods and reinforcement learning methods on two games with very different scales.

Two games are evaluated:

### Tic-Tac-Toe

* Small enough for full minimax and alpha-beta search
* Useful for validating search and RL behaviour in a simple setting

### Connect 4

* Much larger game tree and state space
* Used to expose the limits of full search and simple tabular learning
* Uses depth-limited search with heuristic evaluation for practical play

Experiments compare win rate, draw rate, runtime, and search effort across all required agents.

---

## Features

* Separate game implementations for Tic-Tac-Toe and Connect 4
* Baseline default opponents for both games
* Full minimax and alpha-beta for Tic-Tac-Toe
* Full-search infeasibility benchmark for Connect 4
* Depth-limited minimax and alpha-beta for Connect 4
* Tabular Q-learning for both games
* DQN for both games
* Saved RL model generation
* Full experiment pipeline (CSV output)
* Automated plotting and summary tables
* Interactive Pygame demos for both games

---

## Installation

Python 3.11+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
matplotlib
pandas
pygame
torch
tqdm
```

Optional virtual environment:

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Running the Project

All functionality is controlled from:

```bash
python main.py
```

You will see:

```
1 - Run Tic Tac Toe Pygame
2 - Run Connect 4 Pygame
3 - Run Tic Tac Toe Experiments
4 - Run Connect 4 Experiments
5 - Generate Tic Tac Toe RL Models
6 - Generate Connect 4 RL Models
7 - Build Figures
```

---

## Modes

### 1 — Run Tic Tac Toe Pygame

Launches the Tic-Tac-Toe graphical demo.

* Human vs Human
* Human vs Agent
* Agent vs Agent
* Player 1 and Player 2 mode selection
* Reset support

---

### 2 — Run Connect 4 Pygame

Launches the Connect 4 graphical demo.

* Human vs Human
* Human vs Agent
* Agent vs Agent
* Player 1 and Player 2 mode selection
* Reset support
* Animated token drop

---

### 3 — Run Tic Tac Toe Experiments

Runs automated Tic-Tac-Toe experiments across:

* Default opponent
* Minimax
* Alpha-Beta
* Q-Learning
* DQN

Outputs CSV files to:

```
/results
```

The default run uses 50 games per matchup.

---

### 4 — Run Connect 4 Experiments

Runs automated Connect 4 experiments across:

* Default opponent
* Depth-limited Minimax
* Depth-limited Alpha-Beta
* Q-Learning
* DQN

Outputs CSV files to:

```
/results
```

The default run uses 100 games per matchup.

---

### 5 — Generate Tic Tac Toe RL Models

Generates and saves the Tic-Tac-Toe RL agents:

* Tabular Q-learning table
* DQN model

Saved to:

```
/models
```

---

### 6 — Generate Connect 4 RL Models

Generates and saves the Connect 4 RL agents:

* Tabular Q-learning table
* DQN model

Saved to:

```
/models
```

---

### 7 — Build Figures

Reads saved experiment results and training logs and generates plots and summary CSVs.

Outputs figures to:

```
/figures
```

Outputs summary tables to:

```
/results/analysis
```

Includes:

* Win-rate summaries
* Default-opponent comparisons
* RL training curves
* Connect 4 benchmark/runtime figures
* Search cost summaries

---

## Individual Algorithm Usage

All agent implementations are modular and located in:

```
src/agents/
```

Each can be imported and used independently.

### Tic-Tac-Toe Minimax

```python
from src.agents.tictactoe.minimax import choose_minimax_move
move = choose_minimax_move(game)
```

### Tic-Tac-Toe Alpha-Beta

```python
from src.agents.tictactoe.alphabeta import choose_alphabeta_move
move = choose_alphabeta_move(game)
```

### Tic-Tac-Toe Q-Learning

```python
from src.agents.tictactoe.q_learning import train_q_learning, choose_q_move
q_table = train_q_learning(episodes=50000)
move = choose_q_move(game, q_table)
```

### Tic-Tac-Toe DQN

```python
from src.agents.tictactoe.dqn import train_dqn, choose_dqn_move
model = train_dqn(episodes=50000)
move = choose_dqn_move(game, model)
```

### Connect 4 Depth-Limited Minimax

```python
from src.agents.connect4.minimax import choose_minimax_move_limited
move, stats = choose_minimax_move_limited(game, depth_limit=5)
```

### Connect 4 Depth-Limited Alpha-Beta

```python
from src.agents.connect4.alphabeta import choose_alphabeta_move_limited
move, stats = choose_alphabeta_move_limited(game, depth_limit=5)
```

### Connect 4 Q-Learning

```python
from src.agents.connect4.q_learning import train_q_learning, choose_q_move
q_table = train_q_learning(episodes=50000)
move = choose_q_move(game, q_table)
```

### Connect 4 DQN

```python
from src.agents.connect4.dqn import train_dqn, choose_dqn_move
model = train_dqn(episodes=50000)
move = choose_dqn_move(game, model)
```

---

## Project Structure

```
src/
    agents/
        tictactoe/
            default_opponent.py
            minimax.py
            alphabeta.py
            q_learning.py
            dqn.py
        connect4/
            default_opponent.py
            minimax.py
            alphabeta.py
            q_learning.py
            dqn.py
    games/
        tictactoe/
            game.py
        connect4/
            game.py
    experiments/
        tictactoe_runner.py
        connect4_runner.py
        connect4_search.py
        analysis.py
        training_log.py
    ui/
        tictactoe.py
        connect4.py

figures/            # Generated plots
results/            # CSV outputs and benchmark text
models/             # Saved RL models
samples/            # Demo screenshots / sample outputs
main.py             # Entry point
requirements.txt
README.md
```

---

## Experimental Design

Experiments evaluate:

* Tic-Tac-Toe and Connect 4 separately
* Default opponent comparisons
* Agent-versus-agent comparisons
* Alternating starting player for fairness
* Search runtime and node counts
* RL training curves

Comparisons are made using:

* Win rate
* Draw rate
* Average moves
* Average decision time
* Average nodes visited

For Connect 4 search, the project also includes a benchmark showing that full minimax and full alpha-beta are impractical from the standard starting position.

---

## Submission Contents

The submission ZIP includes:

* All Python source code
* `README.txt`
* `requirements.txt`
* `/results` CSV files
* `/figures` performance plots
* Saved benchmark text
* PDF analysis report
* Demo video

---

## Design Philosophy

* Explicit algorithm implementations
* Separate game-specific logic where needed
* Reproducible experiments
* Clear comparison between search and RL methods
* Full experiment and analysis pipeline
* Simple interactive demos for testing and presentation

---

## License

Educational use only.

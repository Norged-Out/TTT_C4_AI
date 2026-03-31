"""
Author: Priyansh Nayak
Description: Entry point for Tic Tac Toe and Connect 4 project
"""

import csv
import os

from src.experiments.runner import run_experiments


def write_results(filename, results):
    os.makedirs("results", exist_ok=True)

    filepath = os.path.join("results", filename)

    print(f"Total runs for {filename}: {len(results)}")

    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    fieldnames = sorted(all_keys)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {filepath}")


def run_experiment_mode():
    try:
        results = run_experiments(num_games=50)
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")
        return

    write_results("tictactoe_results.csv", results)
    print("Experiment run completed.")


def generate_saved_models():
    try:
        from src.agents.tictactoe.q_learning import Q_TABLE_PATH, train_q_learning
        from src.agents.tictactoe.dqn import DQN_MODEL_PATH, train_dqn
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")
        return

    print("Generating Tic Tac Toe saved models")
    print("Training Q-learning for 20000 episodes")
    train_q_learning(episodes=20000, force_retrain=True)
    print(f"Saved Q-learning table to {Q_TABLE_PATH}")

    print("Training DQN for 20000 episodes")
    train_dqn(episodes=20000, force_retrain=True)
    print(f"Saved DQN model to {DQN_MODEL_PATH}")

    print("Model generation completed.")


def run_pygame_mode():
    try:
        from src.ui.tictactoe import run_game
        run_game()
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")


def run_connect4_pygame_mode():
    try:
        from src.ui.connect4 import run_game
        run_game()
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")


if __name__ == "__main__":
    print("Select Mode:")
    print("1 - Run Tic Tac Toe Pygame")
    print("2 - Run Tic Tac Toe Experiments")
    print("3 - Generate Saved RL Models")
    print("4 - Run Connect 4 Pygame")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        run_pygame_mode()
    elif choice == "2":
        run_experiment_mode()
    elif choice == "3":
        generate_saved_models()
    elif choice == "4":
        run_connect4_pygame_mode()
    else:
        print("Invalid choice.")

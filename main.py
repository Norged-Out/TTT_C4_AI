"""
Author: Priyansh Nayak
Description: Entry point for Tic Tac Toe project
"""

import csv
import os

from src.experiments.runner import run_experiments
from src.ui.tictactoe_text import (
    play_tictactoe_vs_alphabeta,
    play_tictactoe_vs_dqn,
    play_tictactoe,
    play_tictactoe_vs_default,
    play_tictactoe_vs_minimax,
    play_tictactoe_vs_q_learning,
)


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
    results = run_experiments(num_games=50)
    write_results("tictactoe_results.csv", results)
    print("Experiment run completed.")


def run_opponent_menu():
    print("Choose Opponent:")
    print("1 - Default Opponent")
    print("2 - Minimax")
    print("3 - Alpha Beta")
    print("4 - Q-learning")
    print("5 - DQN")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        play_tictactoe_vs_default()
    elif choice == "2":
        play_tictactoe_vs_minimax()
    elif choice == "3":
        play_tictactoe_vs_alphabeta()
    elif choice == "4":
        play_tictactoe_vs_q_learning()
    elif choice == "5":
        play_tictactoe_vs_dqn()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    print("Select Mode:")
    print("1 - Two Player Tic Tac Toe")
    print("2 - Run Tic Tac Toe Experiments")
    print("3 - Choose Opponent")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        play_tictactoe()
    elif choice == "2":
        run_experiment_mode()
    elif choice == "3":
        run_opponent_menu()
    else:
        print("Invalid choice.")

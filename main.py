"""
Author: Priyansh Nayak
Description: Entry point for Tic Tac Toe project
"""

import csv
import os

from src.experiments.runner import run_experiments
from src.ui.tictactoe_text import (
    play_tictactoe_vs_alphabeta,
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


if __name__ == "__main__":
    print("Select Mode:")
    print("1 - Two Player Tic Tac Toe")
    print("2 - Play Against Default Opponent")
    print("3 - Play Against Minimax")
    print("4 - Play Against Alpha Beta")
    print("5 - Run Tic Tac Toe Experiments")
    print("6 - Play Against Q-learning")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        play_tictactoe()
    elif choice == "2":
        play_tictactoe_vs_default()
    elif choice == "3":
        play_tictactoe_vs_minimax()
    elif choice == "4":
        play_tictactoe_vs_alphabeta()
    elif choice == "5":
        run_experiment_mode()
    elif choice == "6":
        play_tictactoe_vs_q_learning()
    else:
        print("Invalid choice.")

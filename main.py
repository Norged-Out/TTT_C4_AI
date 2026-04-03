"""
Author: Priyansh Nayak
Description: Entry point for Tic Tac Toe and Connect 4 project
"""

import csv
import os


def write_results(filename, results):
    # write one csv file for a full experiment block
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


def run_tictactoe_experiment_mode():
    try:
        # Tic Tac Toe uses the lighter experiment set
        from src.experiments.tictactoe_runner import run_experiments

        # run the experiment block
        results = run_experiments(num_games=50)
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")
        return

    write_results("tictactoe_results.csv", results)
    print("Experiment run completed.")


def run_connect4_experiment_mode():
    # Connect 4 uses the larger run count
    from src.experiments.connect4_runner import run_experiments

    # run the experiment block
    results = run_experiments(num_games=10)
    write_results("connect4_results.csv", results)
    print("Connect 4 experiment run completed.")


def generate_tictactoe_models():
    try:
        # generate both saved RL agents together
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


def generate_connect4_models():
    # generate both saved RL agents together
    from src.agents.connect4.q_learning import Q_TABLE_PATH, train_q_learning
    from src.agents.connect4.dqn import DQN_MODEL_PATH, train_dqn

    print("Generating Connect 4 saved models")
    print("Training Q-learning for 50000 episodes")
    train_q_learning(episodes=50000, force_retrain=True)
    print(f"Saved Q-learning table to {Q_TABLE_PATH}")

    print("Training DQN for 50000 episodes")
    train_dqn(episodes=50000, force_retrain=True)
    print(f"Saved DQN model to {DQN_MODEL_PATH}")
    print("Model generation completed.")


def run_tictactoe_pygame_mode():
    try:
        # open the Tic Tac Toe window
        from src.ui.tictactoe import run_game
        run_game()
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")


def run_connect4_pygame_mode():
    try:
        # open the Connect 4 window
        from src.ui.connect4 import run_game
        run_game()
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")



def main():
    # simple menu entry point
    print("Select Mode:")
    print("1 - Run Tic Tac Toe Pygame")
    print("2 - Run Connect 4 Pygame")
    print("3 - Run Tic Tac Toe Experiments")
    print("4 - Run Connect 4 Experiments")
    print("5 - Generate Tic Tac Toe RL Models")
    print("6 - Generate Connect 4 RL Models")

    # one simple input is enough here
    choice = input("Enter choice: ").strip()

    if choice == "1":
        run_tictactoe_pygame_mode()
    elif choice == "2":
        run_connect4_pygame_mode()
    elif choice == "3":
        run_tictactoe_experiment_mode()
    elif choice == "4":
        run_connect4_experiment_mode()
    elif choice == "5":
        generate_tictactoe_models()
    elif choice == "6":
        generate_connect4_models()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()

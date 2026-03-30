"""
Author: Priyansh Nayak
Description: Entry point for Tic Tac Toe project
"""

from src.ui.tictactoe_text import (
    play_tictactoe_vs_alphabeta,
    play_tictactoe,
    play_tictactoe_vs_default,
    play_tictactoe_vs_minimax,
)


if __name__ == "__main__":
    print("Select Mode:")
    print("1 - Two Player Tic Tac Toe")
    print("2 - Play Against Default Opponent")
    print("3 - Play Against Minimax")
    print("4 - Play Against Alpha Beta")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        play_tictactoe()
    elif choice == "2":
        play_tictactoe_vs_default()
    elif choice == "3":
        play_tictactoe_vs_minimax()
    elif choice == "4":
        play_tictactoe_vs_alphabeta()
    else:
        print("Invalid choice.")

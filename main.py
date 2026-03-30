"""
Author: Priyansh Nayak
Description: Entry point for Tic Tac Toe project
"""

from src.ui.tictactoe_text import play_tictactoe, play_tictactoe_vs_default


if __name__ == "__main__":
    print("Select Mode:")
    print("1 - Two Player Tic Tac Toe")
    print("2 - Play Against Default Opponent")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        play_tictactoe()
    elif choice == "2":
        play_tictactoe_vs_default()
    else:
        print("Invalid choice.")
